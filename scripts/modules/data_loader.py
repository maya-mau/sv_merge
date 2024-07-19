from torch.utils.data.dataset import Dataset
from collections import defaultdict
import numpy as np
import torch
import sys

import vcfpy


def get_type_index(record: vcfpy.Record):
    if record.ALT[0].type == "INS":
        return 0
    elif record.ALT[0].type == "DEL":
        return 1
    elif record.ALT[0].type == "DUP":
        return 2
    elif record.ALT[0].type == "INV":
        return 3
    else:
        return 4

type_names = ["INS","DEL","DUP","NA"]


def load_features_from_vcf(
        records: list,
        x: list,
        y: list,
        feature_names: list,
        vcf_path: str,
        truth_info_name: str,
        annotation_name: str,
        filter_fn=None,
        contigs=None
        ):

    print(vcf_path)
    reader = vcfpy.Reader.from_path(vcf_path)

    type_vector = [0,0,0,0,0]

    r = 0
    for record in reader:
        info = record.INFO
        call = record.calls[0]
        if call.data["FT"] != ['PASS']:
            print(call.data["FT"])
            print(record.calls) 
        """ if record.FILTER != ["PASS"]:
            print(record.FILTER)
            print(record) """

        
        if filter_fn is not None:
            if not(filter_fn(record)):
                continue 

        if contigs is not None:
            if record.CHROM not in contigs:
                continue

        records.append(record)

        if record.calls is None:
            exit("ERROR: no calls in record: " + record.ID)
        elif len(record.calls) != 1:
            exit("ERROR: multiple calls in record: " + record.ID)

        # q = 0, p(correct) <= 0.0  Merged with above
        # q = 1, p(correct) <= 0.5  Merged with above
        # q = 2, p(correct) <= 0.75          i = 0 + 1
        # q = 3, p(correct) <= 0.875         i = 1 + 1
        # q = 4, p(correct) <= 0.9375        i = 2 + 1
        # q = 5, p(correct) <= 0.96875       i = 3 + 1
        # q = 6, p(correct) <= 0.984375      i = 4 + 1
        # q = 7, p(correct) <= 0.9921875     i = 5 + 1

        #                     q F NonSpan       q R NonSpan         q F Span         q R Span        is_tandem
        #  Window Depth   |                 |                 |                 |                |  /  length of region evaluated
        #               \  2  3  4  5  6  7  2  3  4  5  6  7  2  3  4  5  6  7  2  3  4  5  6  7  /  /
        #             i 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26
        #             [ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, x]
        
        if truth_info_name.lower() == "hapestry":
            is_true = float(info["HAPESTRY_REF_MAX"]) > 0.9
        elif truth_info_name is not None:
            if truth_info_name not in info:
                sys.stderr.write("ERROR: truth info not found in record: " + str(record.ID) + "\n")
                continue

            is_true = info[truth_info_name]
        else:
            is_true = 0

        ref_length = float(len(record.REF))
        alt_length = float(len(record.ALT[0].serialize()))

        is_tandem = info["tr_coverage"] > 0.9

        t = type_vector
        type_index = get_type_index(record)
        t[type_index] = 1

        caller_support = [0,0,0]
        caller_support[0] = info["SUPP_PAV"] if "SUPP_PAV" in info else 0
        caller_support[1] = info["SUPP_PBSV"] if "SUPP_PBSV" in info else 0
        caller_support[2] = info["SUPP_SNIFFLES"] if "SUPP_SNIFFLES" in info else 0

        hapestry_data = list(map(float,info["HAPESTRY_READS"])) # hapestry_data is HAPESTRY_READS
        hapestry_data_neighbors = float(info["HAPESTRY_READS_NEIGHBORS"]) # hapestry_data_neighbors is HAPESTRY_READS_NEIGHBORS

        hapestry_reads_CCS = info["HAPESTRY_READS_CCS"] if "HAPESTRY_READS_CCS" in info else [] 
        hapestry_reads_max_CCS = info["HAPESTRY_READS_MAX_CCS"] if "HAPESTRY_READS_MAX_CCS" in info else 0
        hapestry_reads_neighbors_CCS = info["HAPESTRY_READS_NEIGHBORS_CCS"] if "HAPESTRY_READS_NEIGHBORS_CCS" in info else 0

        VG_AD = info["VG_AD"] if "VG_AD" in info else [0, 0]
        VG_MAD = info["VG_MAD"] if "VG_MAD" in info else 0
        VG_DP = info["VG_DP"] if "VG_DP" in info else 0
        VG_GL = info["VG_GL"] if "VG_GL" in info else [0, 0, 0]
        VG_GQ = info["VG_GQ"] if "VG_GQ" in info else 0
        
        if "VG_GP" in info: 
            if str(info["VG_GP"]) == 'nan':
                VG_GP = 0
            else:
                VG_GP = info["VG_GP"]
        else:
            VG_GP = 0

        VG_XD = info["VG_XD"] if "VG_XD" in info else 0
        
        GT = list(map(float,call.data["GT"].replace('.','0').split("/"))) if '/' in call.data["GT"] else [0,0]
        OLD_GT = list(map(float,call.data["OLD_GT"].replace('.','0').split("/"))) if '/' in call.data["OLD_GT"] else [0,0]
        FT = [0]
        ADF = call.data['ADF'] if "ADF" in call.data else [0,0]
        ADR = call.data['ADR'] if "ADR" in call.data else [0,0]
        PL = call.data['PL'] if "PL" in call.data else [0,0,0]
            

        y.append(is_true)
        x.append([])

        q_score = record.QUAL if record.QUAL is not None else 0
        stdev_pos = float(info["STDEV_POS"]) if "STDEV_POS" in info else 0
        stdev_len = float(info["STDEV_LEN"]) if "STDEV_LEN" in info else 0


        x[-1].append(q_score)
        if r == 0:
            feature_names.append("q_score")

        x[-1].append(stdev_len)
        if r == 0:
            feature_names.append("stdev_len")

        x[-1].append(stdev_pos)
        if r == 0:
            feature_names.append("stdev_pos")

        x[-1].append(ref_length)
        if r == 0:
            feature_names.append("ref_length")

        x[-1].append(alt_length)
        if r == 0:
            feature_names.append("alt_length")

        x[-1].extend(type_vector)
        if r == 0:
            feature_names.extend(["type_" + str(i) for i in range(len(type_vector))])

        x[-1].extend(caller_support)
        if r == 0:
            feature_names.extend(["caller_support_" + str(i) for i in range(len(caller_support))])

        # add vg features (illumina)
        x[-1].extend(VG_AD)
        if r == 0:
            feature_names.extend(["VG_AD_" + str(i) for i in range(len(VG_AD))])

        x[-1].append(VG_MAD)
        if r == 0:
            feature_names.append("VG_MAD")

        x[-1].append(VG_DP)
        if r == 0:
            feature_names.append("VG_DP") 

        x[-1].extend(VG_GL)
        if r == 0:
            feature_names.extend(["VG_GL_" + str(i) for i in range(len(VG_GL))])

        x[-1].append(VG_GQ)
        if r == 0:
            feature_names.append("VG_GQ") 
            
        x[-1].append(VG_GP)
        if r == 0:
            feature_names.append("VG_GP")

        x[-1].append(VG_XD)
        if r == 0:
            feature_names.append("VG_XD")

        # add paragraph features (illumina)
        x[-1].extend(GT)
        if r == 0:
            feature_names.extend(["GT_" + str(i) for i in range(len(GT))])

        x[-1].extend(OLD_GT)
        if r == 0:
            feature_names.extend(["OLD_GT_" + str(i) for i in range(len(OLD_GT))])

        # FT here




        
        if annotation_name.lower() == "hapestry":
            max_align_score = info["HAPESTRY_READS_MAX"] # is there a reason this is down here?

            # add hapestry features (ONT) OR
            # add hapestry features (PB) if using VG vcfs
            x[-1].extend(hapestry_data)
            if r == 0:
                feature_names.extend(["hapestry_data_" + str(i) for i in range(len(hapestry_data))])

            x[-1].append(hapestry_data_neighbors)
            if r == 0:
                feature_names.append("hapestry_data_neighbors")

            x[-1].append(max_align_score)
            if r == 0:
                feature_names.append("max_align_score") 
            
            # add hapestry_CCS features (PB CCS)
            # hapestry_reads_CCS = hapestry_reads_CCS[:-2] # uncomment only if using both ONT and PB features

            """ x[-1].extend(hapestry_reads_CCS)
            if r == 0:
                feature_names.extend(["hapestry_reads_CCS_" + str(i) for i in range(len(hapestry_reads_CCS))]) 
            
            x[-1].append(hapestry_reads_max_CCS)
            if r == 0:
                feature_names.append("hapestry_reads_max_CCS")
            
            x[-1].append(hapestry_reads_neighbors_CCS)
            if r == 0:
                feature_names.append("happestry_reads_neighbors_CCS") """ 


        elif annotation_name.lower() == "sniffles":
            call = record.calls[0]

            gt = list(map(float,call.data["GT"].replace('.','0').split("/"))) if '/' in call.data["GT"] else [0,0]

            x[-1].extend(gt)
            if r == 0:
                feature_names.extend(["GT_" + str(i) for i in range(len(gt))])

            x[-1].append(call.data["GQ"])
            if r == 0:
                feature_names.append("GQ")

            x[-1].append(call.data["DR"])
            if r == 0:
                feature_names.append("DR")

            x[-1].append(call.data["DV"])
            if r == 0:
                feature_names.append("DV")

            x[-1].append(is_tandem)
            if r == 0:
                feature_names.append("is_tandem")

        elif annotation_name.lower() == "svjedi":
            call = record.calls[0]

            pl = call.data["PL"] if None not in call.data["PL"] else [0,0,0]
            # Sort GT because we don't care about order, just want consistency for the model
            gt = sorted(list(map(float,call.data["GT"].split("|"))) if '|' in call.data["GT"] else [0,0])

            x[-1].extend(gt)
            if r == 0:
                feature_names.extend(["GT_" + str(i) for i in range(len(gt))])

            x[-1].extend(call.data["AD"])
            if r == 0:
                feature_names.extend(["AD_" + str(i) for i in range(len(call.data["AD"]))])

            x[-1].append(call.data["DP"])
            if r == 0:
                feature_names.append("DP")

            x[-1].extend(pl)
            if r == 0:
                feature_names.extend(["PL_" + str(i) for i in range(len(pl))])

            x[-1].append(is_tandem)
            if r == 0:
                feature_names.append("is_tandem")

        elif annotation_name.lower() == "kanpig":
            call = record.calls[0]

            # Sort GT because we don't care about order, just want consistency for the model
            gt = sorted(list(map(float,call.data["GT"].split("|"))) if '|' in call.data["GT"] else [0,0])

            x[-1].extend(gt)
            if r == 0:
                feature_names.extend(["GT_" + str(i) for i in range(len(gt))])

            x[-1].extend(call.data["AD"])
            if r == 0:
                feature_names.extend(["AD_" + str(i) for i in range(len(call.data["AD"]))])

            x[-1].append(call.data["DP"])
            if r == 0:
                feature_names.append("DP")

            x[-1].append(call.data["GQ"])
            if r == 0:
                feature_names.append("GQ")

            x[-1].append(call.data["SQ"])
            if r == 0:
                feature_names.append("SQ")

            x[-1].append(is_tandem)
            if r == 0:
                feature_names.append("is_tandem")

        elif annotation_name.lower() == "cutesv":
            call = record.calls[0]

            # Sort GT because we don't care about order, just want consistency for the model
            gt = sorted(list(map(float,call.data["GT"].split("|"))) if '|' in call.data["GT"] else [0,0])

            x[-1].extend(gt)
            if r == 0:
                feature_names.extend(["GT_" + str(i) for i in range(len(gt))])

            x[-1].append(call.data["GQ"])
            if r == 0:
                feature_names.append("GQ")

            x[-1].append(call.data["DR"])
            if r == 0:
                feature_names.append("DR")

            x[-1].append(call.data["DV"])
            if r == 0:
                feature_names.append("DV")

            x[-1].extend(call.data["PL"])
            if r == 0:
                feature_names.extend(["PL_" + str(i) for i in range(len(call.data["PL"]))])

            x[-1].append(is_tandem)
            if r == 0:
                feature_names.append("is_tandem")

            if r == 0:
                print("GQ", call.data["GQ"])
                print("DR", call.data["DR"])
                print("DV", call.data["DV"])
                print("PL", call.data["PL"])

        elif annotation_name.lower() == "lrcaller":
            call = record.calls[0]

            gt1 = sorted(list(map(float,call.data["GT1"].split("/"))) if '/' in call.data["GT1"] else [0,0])
            gt2 = sorted(list(map(float,call.data["GT2"].split("/"))) if '/' in call.data["GT2"] else [0,0])
            gt3 = sorted(list(map(float,call.data["GT3"].split("/"))) if '/' in call.data["GT3"] else [0,0])
            gt4 = sorted(list(map(float,call.data["GT4"].split("/"))) if '/' in call.data["GT4"] else [0,0])
            gt5 = sorted(list(map(float,call.data["GT5"].split("/"))) if '/' in call.data["GT5"] else [0,0])

            x[-1].extend(gt1)
            if r == 0:
                feature_names.extend(["GT1_" + str(i) for i in range(len(gt1))])

            x[-1].extend(gt2)
            if r == 0:
                feature_names.extend(["GT2_" + str(i) for i in range(len(gt1))])

            x[-1].extend(gt3)
            if r == 0:
                feature_names.extend(["GT3_" + str(i) for i in range(len(gt1))])

            x[-1].extend(gt4)
            if r == 0:
                feature_names.extend(["GT4_" + str(i) for i in range(len(gt1))])

            x[-1].extend(gt5)
            if r == 0:
                feature_names.extend(["GT5_" + str(i) for i in range(len(gt1))])

            x[-1].extend(call.data["AD1"])
            if r == 0:
                feature_names.extend(["AD1_" + str(i) for i in range(len(call.data["AD1"]))])

            x[-1].extend(call.data["AD2"])
            if r == 0:
                feature_names.extend(["AD2_" + str(i) for i in range(len(call.data["AD2"]))])

            x[-1].extend(call.data["AD3"])
            if r == 0:
                feature_names.extend(["AD3_" + str(i) for i in range(len(call.data["AD3"]))])

            x[-1].extend(call.data["AD4"])
            if r == 0:
                feature_names.extend(["AD4_" + str(i) for i in range(len(call.data["AD4"]))])

            x[-1].extend(call.data["AD5"])
            if r == 0:
                feature_names.extend(["AD5_" + str(i) for i in range(len(call.data["AD5"]))])

            x[-1].extend(call.data["VA1"])
            if r == 0:
                feature_names.extend(["VA1_" + str(i) for i in range(len(call.data["VA1"]))])

            x[-1].extend(call.data["VA2"])
            if r == 0:
                feature_names.extend(["VA2_" + str(i) for i in range(len(call.data["VA2"]))])

            x[-1].extend(call.data["VA3"])
            if r == 0:
                feature_names.extend(["VA3_" + str(i) for i in range(len(call.data["VA3"]))])

            x[-1].extend(call.data["VA4"])
            if r == 0:
                feature_names.extend(["VA4_" + str(i) for i in range(len(call.data["VA4"]))])

            x[-1].extend(call.data["VA5"])
            if r == 0:
                feature_names.extend(["VA5_" + str(i) for i in range(len(call.data["VA5"]))])

            x[-1].extend(call.data["PL1"])
            if r == 0:
                feature_names.extend(["PL1_" + str(i) for i in range(len(call.data["PL1"]))])

            x[-1].extend(call.data["PL2"])
            if r == 0:
                feature_names.extend(["PL2_" + str(i) for i in range(len(call.data["PL2"]))])

            x[-1].extend(call.data["PL3"])
            if r == 0:
                feature_names.extend(["PL3_" + str(i) for i in range(len(call.data["PL3"]))])

            x[-1].extend(call.data["PL4"])
            if r == 0:
                feature_names.extend(["PL4_" + str(i) for i in range(len(call.data["PL4"]))])

            x[-1].extend(call.data["PL5"])
            if r == 0:
                feature_names.extend(["PL5_" + str(i) for i in range(len(call.data["PL5"]))])

            x[-1].append(is_tandem)
            if r == 0:
                feature_names.append("is_tandem")
                print("GT1",call.data["GT1"])
                print("GT2",call.data["GT2"])
                print("GT3",call.data["GT3"])
                print("GT4",call.data["GT4"])
                print("GT5",call.data["GT5"])
                print("AD1",call.data["AD1"])
                print("AD2",call.data["AD2"])
                print("AD3",call.data["AD3"])
                print("AD4",call.data["AD4"])
                print("AD5",call.data["AD5"])
                print("VA1",call.data["VA1"])
                print("VA2",call.data["VA2"])
                print("VA3",call.data["VA3"])
                print("VA4",call.data["VA4"])
                print("VA5",call.data["VA5"])
                print("PL1",call.data["PL1"])
                print("PL2",call.data["PL2"])
                print("PL3",call.data["PL3"])
                print("PL4",call.data["PL4"])
                print("PL5",call.data["PL5"])
                print(x[-1])

        if r == 0:
            if len(feature_names) != len(x[-1]):
                print("ERROR: feature names and data length mismatch: names:%d x:%d" % (len(feature_names), len(x[-1])))

        print(feature_names)
        r += 1        


class VcfDataset(Dataset):
    def __init__(self, vcf_paths: list, truth_info_name, annotation_name, batch_size=256, filter_fn=None, contigs=None):
        x = list()
        y = list()

        self.feature_indexes = None
        self.records = list()

        for p,path in enumerate(vcf_paths):
            if len(path) == 0 or path is None:
                continue

            feature_names = list()

            load_features_from_vcf(
                records=self.records,
                x=x,
                y=y,
                vcf_path=path,
                feature_names=feature_names,
                truth_info_name=truth_info_name,
                annotation_name=annotation_name,
                filter_fn=filter_fn,
                contigs=contigs,
            )

            self.feature_indexes = {feature_names[i]: i for i in range(len(feature_names))}

        x = np.array(x) 
        
        y = np.array(y)

        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor

        self.length = x.shape[0]

        self.x_data = torch.from_numpy(x).type(x_dtype)
        self.y_data = torch.from_numpy(y).type(y_dtype)

        self.x_data = torch.nn.functional.normalize(self.x_data, dim=0)
        self.x_data += 1e-12

        self.filter_fn = filter_fn

        self.minority_size = self.compute_minority_size()

        self.ordinals = list(range(self.length))

        self.batch_size = batch_size
        self.subset_ordinals = list()
        self.initialize_iterator()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def compute_minority_size(self):
        unique, counts = np.unique(self.y_data.numpy(), return_counts=True)

        return min(counts)

    def get_n_batches_per_epoch(self):
        return len(self.subset_ordinals) // self.batch_size

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length

    def initialize_iterator(self):
        # Shuffle and iterate over the data, accumulating approximately balanced batches without replacement
        np.random.shuffle(self.ordinals)

        self.subset_ordinals = list()
        accumulated_class_counts = defaultdict(int)

        for i in self.ordinals:
            c = self.y_data[i].item()

            if accumulated_class_counts[c] < self.minority_size:
                self.subset_ordinals.append(i)
                accumulated_class_counts[c] += 1

        np.random.shuffle(self.subset_ordinals)

    def iter_balanced_batches(self, max_batches_per_epoch=sys.maxsize):
        n = 0
        i = 0
        while i + self.batch_size < len(self.subset_ordinals):
            if n >= max_batches_per_epoch:
                break

            batch_indexes = self.subset_ordinals[i:i+self.batch_size]

            yield self.x_data[batch_indexes], self.y_data[batch_indexes]

            i += self.batch_size
            n += 1

        self.initialize_iterator()

    def iter_batches(self, max_batches_per_epoch=sys.maxsize):
        n = 0
        i = 0
        m = 0
        while m < self.length - 1:
            if n >= max_batches_per_epoch:
                break

            m = min(i+self.batch_size, self.length)
            yield self.x_data[i:m], self.y_data[i:m]

            i += self.batch_size
            n += 1
