#include "Sequence.hpp"

#include <stdexcept>

using std::runtime_error;


namespace sv_merge {


Sequence::Sequence(const string& name, const string& sequence):
        name(name),
        sequence(sequence)
{}


StrandedSequence::StrandedSequence(const string& name, const string& sequence, bool is_reverse):
        name(name),
        sequence(sequence),
        is_reverse(is_reverse)
{}


StrandedSequence::StrandedSequence(const string& name, const string& sequence):
        name(name),
        sequence(sequence),
        is_reverse(false)
{}


StrandedSequence::StrandedSequence():
        name(),
        sequence(),
        is_reverse(false)
{}


StrandedQSequence::StrandedQSequence(const string& name, const string& sequence, const vector<uint8_t>& qualities, bool is_reverse):
        name(name),
        sequence(sequence),
        qualities(qualities),
        is_reverse(is_reverse)
{}


StrandedQSequence::StrandedQSequence(const string& name, const string& sequence, const vector<uint8_t>& qualities):
        name(name),
        sequence(sequence),
        qualities(qualities),
        is_reverse(false)
{}


StrandedQSequence::StrandedQSequence(const string& name, const string& sequence, bool is_reverse):
        name(name),
        sequence(sequence),
        is_reverse(is_reverse)
{}


StrandedQSequence::StrandedQSequence(const string& name, const string& sequence):
        name(name),
        sequence(sequence),
        is_reverse(false)
{}


StrandedQSequence::StrandedQSequence():
        name(),
        sequence(),
        is_reverse(false)
{}


char get_complement(char c){
    if (isupper(c)) {
        switch (c){
            case 'A': return 'T';
            case 'C': return 'G';
            case 'G': return 'C';
            case 'T': return 'A';
            case 'N': return 'N';
            default:
                throw runtime_error("ERROR: cannot complement base: " + string(1,c));
        }
    }

    else {
        switch (c){
            case 'a': return 't';
            case 'c': return 'g';
            case 'g': return 'c';
            case 't': return 'a';
            case 'n': return 'n';
            default:
                throw runtime_error("ERROR: cannot complement base: " + string(1,c));
        }
    }
}


void reverse_complement(string& seq){
    string s;

    for (auto iter = seq.rbegin(); iter != seq.rend(); iter++){
        s += get_complement(*iter);
    }

    seq = std::move(s);
}

} // sv_merge
