#include "fetch.hpp"


namespace sv_merge{


void for_each_sample_bam_path(path bam_csv, const function<void(const string& sample_name, const path& bam_path)>& f){
    if (not (bam_csv.extension() == ".csv")){
        throw runtime_error("ERROR: file does not have compatible csv extension: " + bam_csv.string());
    }

    ifstream file(bam_csv);

    if (not (file.is_open() and file.good())){
        throw runtime_error("ERROR: could not read file: " + bam_csv.string());
    }

    char c;
    string sample_name;
    string bam_path;

    int64_t n_delimiters = 0;
    int64_t n_lines = 0;
    int64_t n_char_in_line = 0;
    char delimiter = ',';

    while (file.get(c)){
//        cerr << c << ' ' << sample_name << ' ' << bam_path << '\n';
        if (c == delimiter){
            n_delimiters++;
            continue;
        }
        if (c == '\n'){
            if (n_char_in_line == 0){
                continue;
            }

            if (n_delimiters != 1){
                throw runtime_error("ERROR: incorrect number of delimiters on line: " + to_string(n_lines) + " of file " + bam_csv.string());
            }

            f(sample_name, bam_path);

            sample_name.clear();
            bam_path.clear();
            n_delimiters = 0;
            n_char_in_line = 0;
            n_lines++;
            continue;
        }

        if (n_delimiters == 0){
            sample_name += c;
        }
        else if (n_delimiters == 1){
            bam_path += c;
        }
        else {
            throw runtime_error("ERROR: too many delimiters in bam csv");
        }

        n_char_in_line++;
    }
}


// This is used when only one half of the ref/query dual iterator `for_alignment_in_bam_region` is desired
void null_fn(const CigarInterval &intersection, const interval_t &interval){}


void update_coord(const Alignment& alignment, const CigarInterval& cigar, CigarInterval& coord, bool require_spanning, int32_t ref_start, int32_t ref_stop){
    bool pass = false;

    // Optionally only track coords that intersect the boundaries
    if (require_spanning) {
        pass = (cigar.ref_start == ref_start or cigar.ref_stop == ref_stop);
    }
    else{
        pass = (cigar.ref_start >= ref_start and cigar.ref_stop <= ref_stop);
    }

    // If the alignment is within the ref region, record the query position
    if (pass){
        auto [start,stop] = cigar.get_forward_query_interval();

        if (alignment.is_reverse()){
            if (stop > coord.query_stop){
                coord.query_stop = stop;
                coord.ref_start = cigar.ref_start;
            }
            if (start < coord.query_start){
                coord.query_start = start;
                coord.ref_stop = cigar.ref_stop;
            }
        }
        else{
            if (start < coord.query_start){
                coord.query_start = start;
                coord.ref_start = cigar.ref_start;
            }
            if (stop > coord.query_stop){
                coord.query_stop = stop;
                coord.ref_stop = cigar.ref_stop;
            }
        }
    }
}


/**
 *
 * @param authenticator
 * @param authenticator_mutex
 * @param sample_to_region_reads : must be pre-allocated with sample-->region-->{} (empty vectors) for multithreading
 * @param sample_name
 * @param subregions
 * @param require_spanning : any read that is returned must, among all its alignments, cover the left and right bounds
 * @param bam_path
 */
void extract_subregions_from_sample(
        GoogleAuthenticator& authenticator,
        sample_region_read_map_t& sample_to_region_reads,
        const string& sample_name,
        const vector<Region>& subregions,
        bool require_spanning,
        path bam_path
){
    if (subregions.empty()){
        throw runtime_error("ERROR: subregions empty");
    }

    // Generate a super-region to encompass all subregions, and assume that subregions are sorted, contiguous.
    // If they are not contiguous and sorted, the iterator function will detect that and error out.
    Region super_region;
    super_region.name = subregions[0].name;
    super_region.start = subregions[0].start;
    super_region.stop = subregions.back().stop;

    CigarInterval placeholder;
    placeholder.query_start = numeric_limits<int32_t>::max();
    placeholder.query_stop = numeric_limits<int32_t>::min();
    placeholder.ref_start = numeric_limits<int32_t>::max();
    placeholder.ref_stop = numeric_limits<int32_t>::min();

    // Keep track of the min and max observed query coordinates that intersect the region of interest
    unordered_map <Region, unordered_map<string,CigarInterval> > query_coords_per_region;
    unordered_map<string,string> query_sequences;

    // Keep the F (query) oriented sequence of all the reads if they have an alignment that intersects the region
    unordered_map <Region, unordered_map<string,string> > alignments_per_region;

    // Unused
    vector<interval_t> query_intervals;

    // For this application of directly fetching sequence from the BAM, it doesn't make sense to reinterpret the coords
    // in native/unclipped query sequence space. An error will be thrown if a hardclip is found (see below)
    bool unclip_coords = false;

    // Make sure the system has the necessary authentication env variable to fetch a remote GS URI
    authenticator.try_with_authentication(2, [&](){
        // Iterate each alignment in the ref region
        for_alignment_in_bam_subregions(
            bam_path,
            super_region.to_string(),
            subregions,
            [&](Alignment& alignment, span<const Region>& overlapping_regions){

                if (alignment.is_unmapped() or not alignment.is_primary()){
                    return;
                }

                string name;
                alignment.get_query_name(name);

                // The region of interest is defined in reference coordinate space
                // TODO: stop using dumb for loop for this step, switch to range query
                vector<interval_t> ref_intervals;
                for (auto& r: overlapping_regions){
                    ref_intervals.emplace_back(r.start, r.stop);

                    // Find/or create coord for this region
                    auto& region_coord = query_coords_per_region[r];

                    // Check if this read/query has an existing coord, from a previously iterated supplementary alignment
                    auto result = region_coord.find(name);

                    // If no previous alignment, initialize with the max placeholder
                    if (result == region_coord.end()){
                        // Insert a new placeholder cigar interval and keep a reference to the value inserted
                        result = region_coord.emplace(name, placeholder).first;
                        result->second.is_reverse = alignment.is_reverse();

                        // Try inserting a new empty sequence and keep a reference to the value inserted
                        auto [iter,success] = query_sequences.try_emplace(name,"");

                        // If the value existed already, don't do anything, the sequence has already been extracted
                        if (not success){
                            continue;
                        }

                        auto& x = iter->second;

                        // Fill the value with the sequence
                        alignment.get_query_sequence(x);
                    }
                }

                // Find the widest possible pair of query coordinates which exactly spans the ref region (accounting for DUPs)
                for_cigar_interval_in_alignment(unclip_coords, alignment, ref_intervals, query_intervals,
                    [&](const CigarInterval& intersection, const interval_t& interval) {

                        // This will not catch every instance of a hardclip, but if there is one in the window it will throw
                        if (intersection.is_hardclip()){
                            throw runtime_error("ERROR: query-oriented direct-from-alignment read fetching cannot be accomplished"
                                                " on hardclipped sequences, use whole-BAM coordinate based fetching instead");
                        }

                        // Clips should not be considered to be "spanning" a window bound. This can occur occasionally when
                        // the clip ends at exactly the bound. The adjacent cigar operation should be used instead.
                        if (intersection.is_softclip()){
                            return;
                        }

    //                cerr << cigar_code_to_char[intersection.code] << ' ' << alignment.is_reverse() << " r: " << intersection.ref_start << ',' << intersection.ref_stop << ' ' << "q: " << intersection.query_start << ',' << intersection.query_stop << '\n';

                        // A single alignment may span multiple regions
                        for (auto& region: overlapping_regions){
                            auto& coord = query_coords_per_region.at(region).at(name);

                            update_coord(alignment, intersection, coord, require_spanning, region.start, region.stop);
                        }
                    },
                    null_fn
                );
        });
    });

    // Finally trim the sequences and insert the subsequences into a map which has keys pre-filled
    for (const auto& [region, query_coords]: query_coords_per_region){
        for (const auto& [name, coords]: query_coords){
            bool pass = false;

            if (require_spanning){
                pass = (coords.ref_start == region.start and coords.ref_stop == region.stop);
            }
            else {
                pass = (coords.query_start != placeholder.query_start and coords.query_stop != placeholder.query_stop);
            }

            if (pass) {
                auto i = coords.query_start;
                auto l = coords.query_stop - coords.query_start;

//                cerr << name << ' ' << coords.is_reverse << ' ' << l << ' ' << coords.query_start << ',' << coords.query_stop << '\n';

                if (coords.is_reverse) {
                    auto s = query_sequences[name].substr(i, l);
                    reverse_complement(s);

                    sample_to_region_reads.at(sample_name).at(region).emplace_back(name,s, true);
                }
                else{
                    sample_to_region_reads.at(sample_name).at(region).emplace_back(name,query_sequences[name].substr(i, l), false);
                }
            }
        }
    }
}


/**
 *
 * @param authenticator
 * @param sample_to_region_coords : must be pre-allocated with sample-->region-->{} (empty vectors) for multithreading
 * @param sample_name
 * @param subregions
 * @param require_spanning : any read that is returned must, among all its alignments, cover the left and right bounds
 * @param unclip_coords : reinterpret hardclips as softclips so that query coords are in the native/unclipped sequence
 * @param bam_path
 */
void extract_subregion_coords_from_sample(
        GoogleAuthenticator& authenticator,
        sample_region_coord_map_t& sample_to_region_coords,
        const string& sample_name,
        const vector<Region>& subregions,
        bool require_spanning,
        bool unclip_coords,
        path bam_path
){
    if (subregions.empty()){
        throw runtime_error("ERROR: subregions empty");
    }

    // Generate a super-region to encompass all subregions, and assume that subregions are sorted, contiguous.
    // If they are not contiguous and sorted, the iterator function will detect that and error out.
    Region super_region;
    super_region.name = subregions[0].name;
    super_region.start = subregions[0].start;
    super_region.stop = subregions.back().stop;

    CigarInterval placeholder;
    placeholder.query_start = numeric_limits<int32_t>::max();
    placeholder.query_stop = numeric_limits<int32_t>::min();
    placeholder.ref_start = numeric_limits<int32_t>::max();
    placeholder.ref_stop = numeric_limits<int32_t>::min();

    // Keep track of the min and max observed query coordinates that intersect the region of interest
    unordered_map <Region, unordered_map<string,CigarInterval> > query_coords_per_region;

    // Keep the F (query) oriented sequence of all the reads if they have an alignment that intersects the region
    unordered_map <Region, unordered_map<string,string> > alignments_per_region;

    // Unused
    vector<interval_t> query_intervals;

    // Make sure the system has the necessary authentication env variable to fetch a remote GS URI
    // Mutex is built into this class so it is thread safe
    authenticator.try_with_authentication(2, [&](){
        // Iterate each alignment in the ref region
        for_alignment_in_bam_subregions(
            bam_path,
            super_region.to_string(),
            subregions,
            [&](Alignment& alignment, span<const Region>& overlapping_regions){

                if (alignment.is_unmapped() or not alignment.is_primary()){
                    return;
                }

                string name;
                alignment.get_query_name(name);

                // The region of interest is defined in reference coordinate space
                // TODO: stop using dumb for loop for this step, switch to range query
                vector<interval_t> ref_intervals;
                for (auto& r: overlapping_regions){
                    ref_intervals.emplace_back(r.start, r.stop);

                    // Find/or create coord for this region
                    auto& region_coord = query_coords_per_region[r];

                    // Check if this read/query has an existing coord, from a previously iterated supplementary alignment
                    auto result = region_coord.find(name);

                    // If no previous alignment, initialize with the max placeholder
                    if (result == region_coord.end()){
                        // Insert a new placeholder cigar interval and keep a reference to the value inserted
                        result = region_coord.emplace(name, placeholder).first;
                        result->second.is_reverse = alignment.is_reverse();
                    }
                }

                // Find the widest possible pair of query coordinates which exactly spans the ref region (accounting for DUPs)
                for_cigar_interval_in_alignment(unclip_coords, alignment, ref_intervals, query_intervals,
                    [&](const CigarInterval& intersection, const interval_t& interval) {
                        // Clips should not be considered to be "spanning" a window bound. This can occur occasionally when
                        // the clip ends at exactly the bound. The adjacent cigar operation should be used instead.
                        if (intersection.is_clip()){
                            return;
                        }

    //                    cerr << cigar_code_to_char[intersection.code] << ' ' << alignment.is_reverse() << " r: " << intersection.ref_start << ',' << intersection.ref_stop << ' ' << "q: " << intersection.query_start << ',' << intersection.query_stop << '\n';

                        // A single alignment may span multiple regions
                        for (auto& region: overlapping_regions){
                            auto& coord = query_coords_per_region.at(region).at(name);
                            update_coord(alignment, intersection, coord, require_spanning, region.start, region.stop);
                        }
                    },
                    null_fn
                );
            });

        // Finally trim the sequences and insert the subsequences into a map which has keys pre-filled
        for (const auto& [region, query_coords]: query_coords_per_region){
            for (const auto& [name, coords]: query_coords){
                bool pass = false;

                if (require_spanning){
                    pass = (coords.ref_start == region.start and coords.ref_stop == region.stop);
                }
                else {
                    pass = (coords.query_start != placeholder.query_start and coords.query_stop != placeholder.query_stop);
                }

                if (pass) {
                    sample_to_region_coords.at(sample_name).at(region).emplace_back(name, coords);
                }
            }
        }
    });
}


/**
 * Fetch query coords for each region AND the flank boundaries
 * @param authenticator
 * @param sample_to_region_coords : must be pre-allocated with sample-->region-->{} (empty vectors) for multithreading
 * @param sample_name
 * @param subregions : regions to be fetched by htslib. MUST ALREADY INCLUDE FLANKS.
 * @param require_spanning : any read that is returned must, among all its alignments, cover the left and right bounds
 * @param unclip_coords : reinterpret hardclips as softclips so that query coords are in the native/unclipped sequence
 * @param flank_length : length of flank that will be SUBTRACTED from the ends of regions
 * @param bam_path
 */
void extract_flanked_subregion_coords_from_sample(
        GoogleAuthenticator& authenticator,
        sample_region_flanked_coord_map_t& sample_to_region_coords,
        const string& sample_name,
        const vector<Region>& subregions,
        bool require_spanning,
        bool unclip_coords,
        int32_t flank_length,
        path bam_path
){
    if (subregions.empty()){
        throw runtime_error("ERROR: subregions empty");
    }

    // Generate a super-region to encompass all subregions, and assume that subregions are sorted, contiguous.
    // If they are not contiguous and sorted, the iterator function will detect that and error out.
    Region super_region;
    super_region.name = subregions[0].name;
    super_region.start = subregions[0].start;
    super_region.stop = subregions.back().stop;

    CigarInterval placeholder;
    placeholder.query_start = numeric_limits<int32_t>::max();
    placeholder.query_stop = numeric_limits<int32_t>::min();
    placeholder.ref_start = numeric_limits<int32_t>::max();
    placeholder.ref_stop = numeric_limits<int32_t>::min();

    pair<CigarInterval,CigarInterval> pair_placeholder = {placeholder,placeholder};

    // Keep track of the min and max observed query coordinates that intersect the region of interest
    unordered_map <Region, unordered_map<string,pair<CigarInterval,CigarInterval> > > query_coords_per_region;

    // Keep the F (query) oriented sequence of all the reads if they have an alignment that intersects the region
    unordered_map <Region, unordered_map<string,string> > alignments_per_region;

    // Unused
    vector<interval_t> query_intervals;

    // Make sure the system has the necessary authentication env variable to fetch a remote GS URI
    // Mutex is built into this class so it is thread safe
    authenticator.try_with_authentication(2, [&](){
        // Iterate each alignment in the ref region
        for_alignment_in_bam_subregions(
            bam_path,
            super_region.to_string(),
            subregions,
            [&](Alignment& alignment, span<const Region>& overlapping_regions){

                if (alignment.is_unmapped() or not alignment.is_primary()){
                    return;
                }

                string name;
                alignment.get_query_name(name);

                // The region of interest is defined in reference coordinate space
                // TODO: stop using dumb for loop for this step, switch to range query
                vector<interval_t> ref_intervals;
                for (auto& r: overlapping_regions){
                    // Append the three intervals that account for inner/outer flank boundaries
                    auto inner_left = r.start + flank_length;
                    auto inner_right = r.stop - flank_length;

                    ref_intervals.emplace_back(r.start, inner_left);
                    ref_intervals.emplace_back(inner_left, inner_right);
                    ref_intervals.emplace_back(inner_right, r.stop);

                    if (inner_left > inner_right){
                        throw runtime_error("ERROR: inner left flank bound exceeds inner right flank bound: " + to_string(inner_left) + "," + to_string(inner_right) + " for read " + name);
                    }

                    // Find/or create coords for this region
                    auto& coords = query_coords_per_region[r];

                    // Check if this read/query has an existing coord, from a previously iterated supplementary alignment
                    auto result = coords.find(name);

                    // If no previous alignment, initialize with the max placeholder
                    if (result == coords.end()){
                        // Insert a new placeholder cigar interval and keep a reference to the value inserted
                        result = coords.emplace(name,pair_placeholder).first;
                        result->second.first.is_reverse = alignment.is_reverse();
                        result->second.second.is_reverse = alignment.is_reverse();
                    }
                }

                // Find the widest possible pair of query coordinates which exactly spans the ref region (accounting for DUPs)
                for_cigar_interval_in_alignment(unclip_coords, alignment, ref_intervals, query_intervals,
                    [&](const CigarInterval& intersection, const interval_t& interval) {
                        // Clips should not be considered to be "spanning" a window bound. This can occur occasionally when
                        // the clip ends at exactly the bound. The adjacent cigar operation should be used instead.
                        if (intersection.is_clip()){
                            return;
                        }

//                        cerr << cigar_code_to_char[intersection.code] << ' ' << alignment.is_reverse() << " r: " << intersection.ref_start << ',' << intersection.ref_stop << ' ' << "q: " << intersection.query_start << ',' << intersection.query_stop << '\n';

                        // A single alignment may span multiple regions
                        for (auto& region: overlapping_regions){
                            auto& [inner_coord, outer_coord] = query_coords_per_region.at(region).at(name);

                            // Inner interval should update without require_spanning because it may be a
                            // point element that is never returned by the cigar iterator, but which needs to be
                            // updated anyway
                            update_coord(
                                    alignment,
                                    intersection,
                                    inner_coord,
                                    false,
                                    region.start + flank_length,
                                    region.stop - flank_length
                            );

                            update_coord(
                                    alignment,
                                    intersection,
                                    outer_coord,
                                    require_spanning,
                                    region.start,
                                    region.stop
                            );
                        }
                    },
                    null_fn
                );
            });

        // Finally trim the sequences and insert the subsequences into a map which has keys pre-filled
        for (const auto& [region, query_coords]: query_coords_per_region){
            for (const auto& [name, coords]: query_coords){
                auto& [inner_coord, outer_coord] = coords;
                bool pass = false;

                // Require all four bounds are touched by alignment
                if (require_spanning) {
                    bool inner_pass = (inner_coord.ref_start == region.start + flank_length and inner_coord.ref_stop == region.stop - flank_length);
                    bool outer_pass = (outer_coord.ref_start == region.start and outer_coord.ref_stop == region.stop);
                    pass = (inner_pass and outer_pass);
                }
                else {
                    pass = (outer_coord.query_start != placeholder.query_start and outer_coord.query_stop != placeholder.query_stop);
                }

                if (pass) {
                    pair<CigarInterval,CigarInterval> c = {inner_coord, outer_coord};
                    sample_to_region_coords.at(sample_name).at(region).emplace_back(name, c);
                }
            }
        }
    });
}


void extract_subsequences_from_sample_thread_fn(
        GoogleAuthenticator& authenticator,
        sample_region_read_map_t & sample_to_region_reads,
        const vector <pair <string,path> >& sample_bams,
        const vector<Region>& regions,
        bool require_spanning,
        atomic<size_t>& job_index
){

    size_t i = job_index.fetch_add(1);

    while (i < sample_bams.size()){
        const auto& [sample_name, bam_path] = sample_bams[i];

        Timer t;

        extract_subregions_from_sample(
                authenticator,
                sample_to_region_reads,
                sample_name,
                regions,
                require_spanning,
                bam_path
        );

        cerr << t << "Elapsed for: " << sample_name << '\n';

        i = job_index.fetch_add(1);
    }
}


void extract_subregion_coords_from_sample_thread_fn(
        GoogleAuthenticator& authenticator,
        sample_region_flanked_coord_map_t& sample_to_region_coords,
        const vector <pair <string,path> >& sample_bams,
        const vector<Region>& regions,
        bool require_spanning,
        int32_t flank_length,
        atomic<size_t>& job_index
){
    size_t i = job_index.fetch_add(1);

    while (i < sample_bams.size()){
        const auto& [sample_name, bam_path] = sample_bams[i];

        Timer t;

        extract_flanked_subregion_coords_from_sample(
                authenticator,
                sample_to_region_coords,
                sample_name,
                regions,
                require_spanning,
                true,
                flank_length,
                bam_path
        );

        cerr << t << "Elapsed for: " << sample_name << '\n';

        i = job_index.fetch_add(1);
    }
}


void get_reads_for_each_bam_subregion(
        Timer& t,
        vector<Region>& regions,
        GoogleAuthenticator& authenticator,
        sample_region_read_map_t& sample_to_region_reads,
        path bam_csv,
        int64_t n_threads,
        bool require_spanning
){
    // Intermediate objects
    vector <pair <string, path> > sample_bams;
    TransMap sample_only_transmap;

    cerr << t << "Loading CSV" << '\n';

    // Load BAM paths as a map with sample->bam
    for_each_sample_bam_path(bam_csv, [&](const string& sample_name, const path& bam_path){
        sample_only_transmap.add_sample(sample_name);
        sample_bams.emplace_back(sample_name, bam_path);

        // Initialize every combo of sample,region with an empty vector
        for (const auto& region: regions){
            sample_to_region_reads[sample_name][region] = {};
        }
    });

    cerr << t << "Processing windows" << '\n';

    // Thread-related variables
    atomic<size_t> job_index = 0;
    vector<thread> threads;

    threads.reserve(n_threads);

    // Launch threads
    for (uint64_t n=0; n<n_threads; n++){
        try {
            cerr << "launching: " << n << '\n';
            threads.emplace_back(
                    std::ref(extract_subsequences_from_sample_thread_fn),
                    std::ref(authenticator),
                    std::ref(sample_to_region_reads),
                    std::cref(sample_bams),
                    std::cref(regions),
                    std::ref(require_spanning),
                    std::ref(job_index)
            );
        } catch (const exception& e) {
            throw e;
        }
    }

    // Wait for threads to finish
    for (auto& n: threads){
        n.join();
    }

}


void get_read_coords_for_each_bam_subregion(
        Timer& t,
        vector<Region>& regions,
        GoogleAuthenticator& authenticator,
        sample_region_flanked_coord_map_t& sample_to_region_coords,
        path bam_csv,
        int64_t n_threads,
        int32_t flank_length,
        bool require_spanning
){
    // Intermediate objects
    vector <pair <string, path> > sample_bams;
    TransMap sample_only_transmap;

    cerr << t << "Loading CSV: " << bam_csv.string() << '\n';

    // Load BAM paths as a map with sample->bam
    for_each_sample_bam_path(bam_csv, [&](const string& sample_name, const path& bam_path){
        sample_only_transmap.add_sample(sample_name);
        sample_bams.emplace_back(sample_name, bam_path);

        // Initialize every combo of sample,region with an empty vector
        for (const auto& region: regions){
            sample_to_region_coords[sample_name][region] = {};
        }
    });

    cerr << t << "Processing windows" << '\n';

    // Thread-related variables
    atomic<size_t> job_index = 0;
    vector<thread> threads;

    threads.reserve(n_threads);

    // Launch threads
    for (uint64_t n=0; n<n_threads; n++){
        try {
            cerr << "launching: " << n << '\n';
            threads.emplace_back(
                    std::ref(extract_subregion_coords_from_sample_thread_fn),
                    std::ref(authenticator),
                    std::ref(sample_to_region_coords),
                    std::cref(sample_bams),
                    std::cref(regions),
                    std::ref(require_spanning),
                    flank_length,
                    std::ref(job_index)
            );
        } catch (const exception& e) {
            throw e;
        }
    }

    // Wait for threads to finish
    for (auto& n: threads){
        n.join();
    }

}


void fetch_reads(
        Timer& t,
        vector<Region>& regions,
        path bam_csv,
        int64_t n_threads,
        bool require_spanning,
        unordered_map<Region,TransMap>& region_transmaps,
        bool append_sample_to_read
){
    GoogleAuthenticator authenticator;
    TransMap template_transmap;

    // Intermediate object to store results of multithreaded sample read fetching
    sample_region_read_map_t sample_to_region_reads;

    // Get a nested map of sample -> region -> vector<Sequence>
    get_reads_for_each_bam_subregion(
            t,
            regions,
            authenticator,
            sample_to_region_reads,
            bam_csv,
            n_threads,
            require_spanning
    );

    // Construct template transmap with only samples
    for (const auto& [sample_name,item]: sample_to_region_reads){
        template_transmap.add_sample(sample_name);
    }

    // Compute coverages for regions
    unordered_map<Region, size_t> region_coverage;
    for (const auto& [sample_name,item]: sample_to_region_reads) {
        for (const auto& [region,sequences]: item) {
            region_coverage[region] += sequences.size();
        }
    }

    // Copy the template transmap into every region and reserve approximate space for nodes/edges/sequences
    region_transmaps.reserve(regions.size());
    for (const auto& r: regions){
        auto item = region_transmaps.emplace(r, template_transmap).first->second;
        auto n_reads = region_coverage[r];

        // Number of reads is known exactly
        item.reserve_sequences(n_reads + 2);
    }

    // Move the downloaded data into a transmap, construct edges for sample->read
    for (auto& [sample_name,item]: sample_to_region_reads){
        for (auto& [region,sequences]: item){
            auto& transmap = region_transmaps.at(region);

            auto sample_id = transmap.get_id(sample_name);

            for (auto& s: sequences) {
                // If the user wants, we append sample name to the read name to prevent intersample collisions
                if (append_sample_to_read) {
                    s.name += + "_" + sample_name;
                }

                transmap.add_read_with_move(s.name, s.sequence, s.is_reverse);
                transmap.add_edge(sample_id, transmap.get_id(s.name), 0);
            }
        }
    }
}


void fetch_query_seqs_for_each_sample_thread_fn(
        const vector <pair <string,path> >& sample_bams,
        unordered_map<string, unordered_map<string,string> >& sample_queries,
        GoogleAuthenticator& authenticator,
        mutex& authenticator_mutex,
        atomic<size_t>& job_index
){

    size_t i = job_index.fetch_add(1);

    while (i < sample_bams.size()){
        Timer t;

        // Fetch the next sample in the job list and break out some vars for that sample
        const auto& [sample_name, bam_path] = sample_bams.at(i);
        auto& queries = sample_queries.at(sample_name);

        // Make sure the system has the necessary authentication env variable to fetch a remote GS URI
        authenticator.try_with_authentication(2, [&]() {
            for_alignment_in_bam(bam_path, [&](Alignment& alignment) {
                // The goal is to collect all query sequences, so skip any that may be hardclipped (not primary)
                if ((not alignment.is_primary()) or alignment.is_supplementary()) {
                    return;
                }

                string name;
                alignment.get_query_name(name);

                auto result = queries.find(name);
                if (result == queries.end()) {
                    return;
                }

                // Fetch the sequence, filling it in directly to the `queries` result object which has been pre-allocated for
                // thread safety
                alignment.get_query_sequence(result->second);
            });
        });

        i = job_index.fetch_add(1);

        cerr << t << "Elapsed for: " << sample_name << '\n';
    }
}


void fetch_query_seqs_for_each_sample(
        path bam_csv,
        int64_t n_threads,
        GoogleAuthenticator& authenticator,
        unordered_map<string, unordered_map<string,string> >& sample_queries
){
    vector <pair <string,path> > sample_bams;

    // Load BAM paths as a map with sample->bam
    for_each_sample_bam_path(bam_csv, [&](const string& sample_name, const path& bam_path){
        sample_bams.emplace_back(sample_name, bam_path);
    });

    // Thread-related variables
    atomic<size_t> job_index = 0;
    vector<thread> threads;
    mutex authenticator_mutex;

    threads.reserve(n_threads);

    // Launch threads
    for (uint64_t n=0; n<n_threads; n++){
        try {
            cerr << "launching: " << n << '\n';
            threads.emplace_back(
                    std::ref(fetch_query_seqs_for_each_sample_thread_fn),
                    std::cref(sample_bams),
                    std::ref(sample_queries),
                    std::ref(authenticator),
                    std::ref(authenticator_mutex),
                    std::ref(job_index)
            );
        } catch (const exception& e) {
            throw e;
        }
    }

    // Wait for threads to finish
    for (auto& n: threads){
        n.join();
    }

}


void fetch_reads_from_clipped_bam(
        Timer& t,
        vector<Region>& regions,
        path bam_csv,
        int64_t n_threads,
        int32_t interval_max_length,
        int32_t flank_length,
        bool require_spanning,
        unordered_map<Region,TransMap>& region_transmaps,
        bool append_sample_to_read
){
    GoogleAuthenticator authenticator;
    TransMap template_transmap;

    // Intermediate object to store results of multithreaded sample read fetching
    sample_region_flanked_coord_map_t sample_to_region_coords;

    // Get a nested map of sample -> region -> vector<Sequence>
    get_read_coords_for_each_bam_subregion(
            t,
            regions,
            authenticator,
            sample_to_region_coords,
            bam_csv,
            n_threads,
            flank_length,
            require_spanning
    );

    // Construct template transmap with only samples
    for (const auto& [sample_name,item]: sample_to_region_coords){
        template_transmap.add_sample(sample_name);
    }

    // Collect all query names in one place and construct thread-safe container for results of fetching
    unordered_map<string, unordered_map<string,string> > sample_queries;
    for (const auto& [sample_name,item]: sample_to_region_coords) {
        // Make sure there is at least a placeholder for samples, so there is no error later. They can be empty.
        sample_queries[sample_name] = {};

        for (const auto& [region,named_coords]: item) {
            for (const auto& [name,coord]: named_coords){
                sample_queries[sample_name].emplace(name,"");
            }
        }
    }

    fetch_query_seqs_for_each_sample(
            bam_csv,
            n_threads,
            authenticator,
            sample_queries
    );

    // Compute coverages for regions
    unordered_map<Region, size_t> region_coverage;
    for (const auto& [sample_name,item]: sample_to_region_coords) {
        for (const auto& [region,coords]: item) {
            region_coverage[region] += coords.size();
        }
    }

    // Copy the template transmap into every region and reserve approximate space for nodes/edges/sequences
    region_transmaps.reserve(regions.size());
    for (const auto& r: regions){
        auto item = region_transmaps.emplace(r, template_transmap).first->second;
        auto n_reads = region_coverage[r];

        // Number of reads is known exactly
        item.reserve_sequences(n_reads + 2);
    }

    auto max_hap_length = size_t(float(interval_max_length) * 2.5);

    // Move the downloaded data into a transmap, construct edges for sample->read
    for (auto& [sample_name,item]: sample_to_region_coords){
        for (auto& [region,named_coords]: item){
            auto& transmap = region_transmaps.at(region);

            auto sample_id = transmap.get_id(sample_name);

            for (auto& [name,coords]: named_coords) {
                auto& [inner_coord, outer_coord] = coords;

                auto i = size_t(outer_coord.query_start);
                auto l = size_t(outer_coord.query_stop - outer_coord.query_start);

                if (l > max_hap_length){
                    cerr << "Warning: skipping reference haplotype " + name + " longer than " + to_string(max_hap_length) + " in window " + region.to_string() << '\n';
                    continue;
                }

                const auto& seq = sample_queries.at(sample_name).at(name);

                if (seq.empty() or (i > seq.size() - 1) or (i + l > seq.size())){
                    throw runtime_error("ERROR: fetch_reads_from_clipped_bam coord slice attempted on empty or undersized sequence: \n" +
                                        sample_name + ',' + name + ",size=" + to_string(seq.size()) + ',' + to_string(i) + ',' + to_string(i+l));
                }

                string s = seq.substr(i, l);;

                inner_coord.query_start -= outer_coord.query_start;
                inner_coord.query_stop -= outer_coord.query_start;

                if (outer_coord.is_reverse) {
                    reverse_complement(s);

                    // Reorient inner coordinates also
                    inner_coord.query_start = int32_t(l) - inner_coord.query_start;
                    inner_coord.query_stop = int32_t(l) - inner_coord.query_stop;
                    std::swap(inner_coord.query_start, inner_coord.query_stop);
                }

                // If the user wants, we append sample name to the read name to prevent intersample collisions
                if (append_sample_to_read) {
                    name += + "_" + sample_name;
                }

                // Finally update the transmap
                transmap.add_read_with_move(name, s);
                transmap.add_edge(sample_id, transmap.get_id(name), 0);
                transmap.add_flank_coord(name, inner_coord.query_start, inner_coord.query_stop);

                // If there are multiple sequences, only choose the first sequence arbitrarily
                break;
            }
        }
    }
}


}
