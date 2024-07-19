import time
start = time.time()


from modules.shallow_linear import ShallowLinear 
from modules.data_loader import VcfDataset
from collections import defaultdict
from torch import multiprocessing

# from modules.misc import plot_roc_curve, write_recalibrated_vcf, plot_confusion

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import torch.nn as nn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

from matplotlib import pyplot

import numpy as np

from datetime import datetime
from collections import OrderedDict
from copy import deepcopy
import random
import numpy
import vcfpy
import torch
import math
import sys
import os

# set matplotlib to use the Agg backend
pyplot.switch_backend('Agg')

# set number of threads for pytorch to 32
n_threads = 32
torch.set_num_threads(n_threads)

print("USING pytorch VERSION: ", torch.__version__)


def train(model, train_loader, test_loader, optimizer, scheduler, loss_fn, epochs, max_batches_per_epoch):
    train_losses = list()
    test_losses = list()
    test_aucs = list()
    models = list()

    model.eval()
    y_predict, y_true, test_loss = test(model=model, loader=test_loader, loss_fn=loss_fn)
    test_losses.append(test_loss)
    test_aucs.append(roc_auc_score(y_true=y_true, y_score=y_predict))
    models.append(model.state_dict())
    model.train()

    n_total_positive = 0
    n_total_negative = 0

    prev_model = None

    batch_index = 0
    for e in range(epochs):
        for i, (x, y) in enumerate(train_loader.iter_balanced_batches(max_batches_per_epoch=max_batches_per_epoch)):
            if i >= max_batches_per_epoch:
                break

            loss = train_batch(model=model, x=x, y=y, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn)

            n_positive = sum(y.data.numpy())
            n_negative = len(y) - n_positive

            n_total_positive += n_positive
            n_total_negative += n_negative

            train_losses.append(loss)

            batch_index += 1

        if e % 1 == 0:
            model.eval()
            y_predict, y_true, test_loss = test(model=model, loader=test_loader, loss_fn=torch.nn.BCELoss(), use_sigmoid=True)

            auc = roc_auc_score(y_true=y_true, y_score=y_predict)

            test_losses.append(test_loss)
            test_aucs.append(auc)

            models.append(deepcopy(model.state_dict()))
            model.train()

            if len(test_losses) > 1:
                test_loss_worsened = test_losses[-1] > test_losses[-2]
                test_auc_worsened = test_aucs[-1] < test_aucs[-2]

                print(f"Epoch:{e}\tbatches:{batch_index}\tlr_rate:{scheduler.get_last_lr()}\tn_positive:{n_total_positive}\tn_negative:{n_total_negative}\tprev_loss:{test_losses[-2]:.3f}\tloss:{test_losses[-1]:.3f}\tworsened:{test_loss_worsened}\tprev_auc:{test_aucs[-2]:.3f}\tauc:{test_aucs[-1]:.3f}\tworsened:{test_auc_worsened}")
    # Get index of greatest AUC
    max_auc_index = test_aucs.index(max(test_aucs))

    # Get the model with the greatest AUC
    model.load_state_dict(models[max_auc_index])

    return train_losses, test_losses, test_aucs


def train_batch(model, x, y, optimizer, scheduler, loss_fn):
    # Run forward calculation
    y_predict = model.forward(x)

    # convert 1-hot vectors back into indices
    # max_values, target_index = y.max(dim=1)
    # target_index = target_index.type(torch.LongTensor)

    # Compute loss.
    loss = loss_fn(y_predict.squeeze(), y)

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    # Update the learning rate
    scheduler.step()

    return loss.data.item()


def test(model, loader, loss_fn, use_sigmoid=False):
    y_vectors = list()
    y_predict_vectors = list()
    losses = list()

    batch_index = 0
    for x, y in loader.iter_batches():
        y_predict = model.forward(x, use_sigmoid=use_sigmoid)

        y_vectors.append(y.data.numpy())
        y_predict_vectors.append(y_predict.data.numpy())

        batch_index += 1

        loss = loss_fn(y_predict.squeeze(), y)
        losses.append(loss.data.numpy())

    y_predict_vector = np.concatenate(y_predict_vectors)
    y_vector = np.concatenate(y_vectors)

    loss = numpy.sum(losses)

    return y_predict_vector, y_vector, loss


def plot_loss(train_losses, test_losses, test_aucs, label, output_dir=None):
    fig,ax = pyplot.subplots(nrows=3)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Train Loss")
    ax[1].set_ylabel("Test Loss")
    ax[2].set_ylabel("Test AUC")

    x_train = list(range(len(train_losses)))
    x_test = list(range(len(test_losses)))

    ax[0].plot(x_train, train_losses, label="train")
    ax[1].plot(x_test, test_losses, label="test")
    ax[2].plot(x_test, test_aucs, label="test")

    fig.suptitle(label)

    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, label + "_loss.png"), dpi=200)


def run(dataset_train, dataset_test, output_dir, downsample=False):
    # Define the hyperparameters
    learning_rate_max = 1
    learning_rate_base = 1e-5
    weight_decay = 4e-6
    dropout_rate = 0.01

    goal_batch_size = 4096
    n_epochs = 16

    config_path = os.path.join(output_dir, "config.txt")

    # Batch size is the number of training examples used to calculate each iteration's gradient
    batch_size_train = min(goal_batch_size, len(dataset_train))

    max_batches_per_epoch = sys.maxsize

    # write the hyperparameters to a file
    with open(config_path, "w") as f:
        f.write("learning_rate_base: {}\n".format(learning_rate_base))
        f.write("learning_rate_max: {}\n".format(learning_rate_max))
        f.write("weight_decay: {}\n".format(weight_decay))
        f.write("n_epochs: {}\n".format(n_epochs))
        f.write("goal_batch_size: {}\n".format(goal_batch_size))
        f.write("downsample: {}\n".format(downsample))
        f.write("batch_size_train: {}\n".format(batch_size_train))
        f.write("dropout_rate: {}\n".format(dropout_rate))
        if dataset_train.filter_fn is not None:
            f.write("filter_fn: {}\n".format(str(dataset_train.filter_fn.__name__)))
        else:
            f.write("filter_fn: None\n")

    input_size = len(dataset_train[0][0])
    print("using input size: ", input_size)
    shallow_model = ShallowLinear(input_size=input_size, dropout_rate=dropout_rate)

    # Initialize the optimizer with above parameters
    optimizer = optim.SGD(shallow_model.parameters(), lr=learning_rate_base, weight_decay=weight_decay)

    # Initialize CLR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate_max, steps_per_epoch=dataset_train.get_n_batches_per_epoch(), epochs=n_epochs)

    # Define the loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # Train and get the resulting loss per iteration  
    train_losses, test_losses, test_aucs = train(model=shallow_model, train_loader=dataset_train, test_loader=dataset_test, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, epochs=n_epochs, max_batches_per_epoch=max_batches_per_epoch)

    # Test and get the resulting predicted y values
    shallow_model.eval()    # switch to eval mode to disable dropout

    return train_losses, test_losses, test_aucs, shallow_model 


def compute_downsampling_weight_tensor(dataset):
    y_data = dataset.y_data

    n_total = len(y_data)
    n_true = sum(y_data)
    n_false = n_total - sum(y_data)
    weight_tensor = torch.clone(y_data).detach()
    # weight = None

    weight_tensor[weight_tensor == 1] = 1/n_true
    weight_tensor[weight_tensor == 0] = 1/n_false

    expected_epoch_size = float(torch.sum(weight_tensor))

    print("n_true: %d\tn_false: %d\texpected_epoch_size: %d" % (n_true, n_false, expected_epoch_size))

    return weight_tensor


'''
Downsamples the test data to have an equal number of true and false labels. Is a hard downsample, used in testing,
whereas the training data is downsampled using weighted random sampling.
'''
def downsample_test_data(y_true, y_predict, seed=37):
    random.seed(seed)

    n_true = sum(y_true)
    n_false = len(y_true) - n_true

    true_weight = min(1.0, float(n_false) / float(n_true))
    false_weight = min(1.0, float(n_true) / float(n_false))

    mask = numpy.zeros(len(y_true))

    for i in range(len(y_true)):
        if y_true[i] == 1 and random.random() <= true_weight:
            mask[i] = 1
        if y_true[i] == 0 and random.random() <= false_weight:
            mask[i] = 1

    y_true = y_true[mask == 1]
    y_predict = y_predict[mask == 1]

    return y_true, y_predict


def plot_roc_curve(y_true, y_predict, label, axes, color=None, alpha=1.0, style='-', use_text=False):
    fpr, tpr, thresholds = roc_curve(y_true, y_predict)
    auc = roc_auc_score(y_true, y_predict) 
    labels_with_auc = []

    if color is not None:
        if label is not None:
            label_uncropped = label + " (AUC " + str(auc) + ")" # for txt file
            label = label + " (AUC " + str(round(auc, 3)) + ")" # for plot
            labels_with_auc.append(label_uncropped)
            axes.plot(fpr, tpr, label=label, color=color, linestyle=style, alpha=alpha)
        else:
            axes.plot(fpr, tpr, color=color, linestyle=style, alpha=alpha)

    else:
        if label is not None:
            label_uncropped = label + " (AUC " + str(auc) + ")" # for txt file
            label = label + " (AUC " + str(round(auc, 3)) + ")" # for plot
            labels_with_auc.append(label_uncropped)
            axes.plot(fpr, tpr, label=label, linestyle=style, alpha=alpha)
        else:
            axes.plot(fpr, tpr, linestyle=style, alpha=alpha)

        color = axes.get_lines()[-1].get_color()

    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")

    axes.grid(which='major', color='black', linestyle='-', linewidth=0.5)
    axes.grid(which='minor', color='black', linestyle=':', linewidth=0.5)

    if use_text:
        l = len(thresholds)
        n = 10
        interval = max(1,(l//n))

        for i,[x,y,t] in enumerate(zip(fpr, tpr, thresholds)):
            # only label 1/10 of the points
            if i % interval == 0:
                axes.text(x, y, round(t, 2), fontsize='small')
                # add marker
                axes.plot(x, y, marker='o', color=color, markeredgewidth=0)

    return axes, fpr, tpr, thresholds, labels_with_auc

def value_labels_on_bar(rects, ax):
    # puts a text label with the bar's value above each bar
    for rect in rects: 
        x = rect.get_x() + rect.get_width() / 2
        y = rect.get_height()
        value = rect.get_height()
        if y < 0.5:
            y = 0.5
        elif y == 1:
            y = 0.97
        
        ax.text(x, y, str(value)[:5], ha='center', va='bottom')

def plot_auc_bar(y_true, y_predict, label, axes, color=None, alpha=1.0):
    # calculates AUC value
    auc = roc_auc_score(y_true, y_predict) 
    # interesting, maya check this out: y_true and/or y_predict is a list TODO
    label_new = label.replace(" ", "\n").replace("NON-TANDEM", "NT").replace("TANDEM", "T")

    if color is not None: 
        rect = axes.bar(x=label_new, height=auc, color=color)
        value_labels_on_bar(rect, axes)

    else:
        rect = axes.bar(x=label_new, height=auc)
        value_labels_on_bar(rect, axes)
        color = axes.get_lines()[-1].get_color()

    axes.set_xlabel("Category")
    axes.set_ylabel("AUC")

    return axes


def write_filtered_vcf(y_predict, records, threshold, input_vcf_path, output_vcf_path):
    reader = vcfpy.Reader.from_path(input_vcf_path)

    writer = vcfpy.Writer.from_path(output_vcf_path, reader.header)

    for record, score in zip(records, y_predict):
        if score > threshold:
            writer.write_record(record)


def min50bp(record):
    if type(record.INFO["SVLEN"]) == list:
        return record.INFO["SVLEN"][0] >= 50
    else:
        return record.INFO["SVLEN"] >= 50
    
# maya's added function, returns true or false 
def min1000bp(record):
    if type(record.INFO["SVLEN"]) == list:
        return record.INFO["SVLEN"][0] >= 1000
    else:
        return record.INFO["SVLEN"][0] >= 1000


def thread_fn(truth_info_name, annotation_name, train_vcfs, train_contigs, test_vcfs, test_contigs, eval_vcfs, eval_contigs, filter_fn, output_dir):
    print("Thread started with ", truth_info_name, annotation_name)

    label = truth_info_name + " truth labels and " + annotation_name + " features"

    dataset_train = VcfDataset(vcf_paths=train_vcfs, truth_info_name=truth_info_name, annotation_name=annotation_name, contigs=train_contigs, filter_fn=filter_fn)
    dataset_test = VcfDataset(vcf_paths=test_vcfs, truth_info_name=truth_info_name, annotation_name=annotation_name, contigs=test_contigs, filter_fn=filter_fn)
    dataset_eval = VcfDataset(vcf_paths=eval_vcfs, truth_info_name=truth_info_name, annotation_name=annotation_name, contigs=eval_contigs, filter_fn=filter_fn)

    n_train = len(dataset_train)
    n_test = len(dataset_test)

    # Print information on the train and test datasets
    print('Train datapoints: ', n_train)
    print('Test datapoints: ', n_test)
    print('Input shape: ', dataset_train[0][0].shape)
    print('Output shape: ', dataset_train[0][1].shape)

    train_losses, test_losses, test_aucs, model = run(dataset_train=dataset_train, dataset_test=dataset_test, downsample=True, output_dir=output_dir)

    # Format the date and time
    model_output_path = os.path.join(output_dir, label.replace(" ", "_") + ".pt")
    torch.save(model.state_dict(), model_output_path)

    # calculate and plot loss
    print("Final loss:", sum(train_losses[-100:])/100)
    plot_loss(train_losses=train_losses, test_losses=test_losses, test_aucs=test_aucs, label=label, output_dir=output_dir)

    x = dataset_eval.x_data.numpy()

    # Finally evaluate on a 3rd dataset which is not used to select training termination conditions
    # Need to use_sigmoid this time to get meaningful 0-1 values
    y_predict, y_true, mean_loss = test(model=model, loader=dataset_eval, loss_fn=torch.nn.BCELoss(), use_sigmoid=True)

    return dataset_eval.records, x, dataset_eval.feature_indexes, y_true, y_predict, truth_info_name, annotation_name


def plot_tandem_stratified_roc_curves(axes, records, y_true, y_predict, truth_info_name, annotation_name):
    tandem = [[],[]]
    non_tandem = [[],[]]
    labels_with_auc = ["Tandem stratified ROC curves"]

    for r,record in enumerate(records):
        is_tandem = record.INFO["tr_coverage"] > 0.9

        if is_tandem:
            tandem[0].append(y_true[r])
            tandem[1].append(y_predict[r])
        else:
            non_tandem[0].append(y_true[r])
            non_tandem[1].append(y_predict[r])

    label = truth_info_name + " truth labels and " + annotation_name + " features TANDEM"
    axes, fpr, tpr, thresholds, labels = plot_roc_curve(y_true=tandem[0], y_predict=tandem[1], label=label, axes=axes, style=':')
    labels_with_auc.extend(labels)

    color = axes.get_lines()[-1].get_color()

    label = truth_info_name + " truth labels and " + annotation_name + " features NON-TANDEM"
    axes, fpr, tpr, thresholds, labels = plot_roc_curve(y_true=non_tandem[0], y_predict=non_tandem[1], label=label, color=color, axes=axes)
    labels_with_auc.extend(labels)

    return axes, labels_with_auc


def get_length(record: vcfpy.Record):
    if "SVLEN" in record.INFO:
        if type(record.INFO["SVLEN"]) == list:
            return abs(record.INFO["SVLEN"][0])
        else:
            return abs(record.INFO["SVLEN"])
    else:
        return abs(len(record.ALT[0]) - len(record.REF))


def plot_length_stratified_roc_curves(axes, records, y_true, y_predict, truth_info_name, annotation_name, tandem_only=False):
    length_stratified_results = defaultdict(lambda: [[],[]])
    colormap = pyplot.colormaps['viridis']
    labels_with_auc = ["Length stratified ROC curves"]

    for r,record in enumerate(records):
        length_bin = int(math.log2(get_length(record) + 1))

        if tandem_only and record.INFO["tr_coverage"] <= 0.9:
            continue 

        length_stratified_results[length_bin][0].append(y_true[r])
        length_stratified_results[length_bin][1].append(y_predict[r])

    for i,(l,result) in enumerate(sorted(length_stratified_results.items(), key=lambda x: x[0])):
        bin_start = 2**(l)
        bin_end = 2**(l+1)
        label = truth_info_name + " truth labels and " + annotation_name + " features " + str(bin_start) + " to " + str(bin_end) + " bp"

        f = float(i)/float(len(length_stratified_results))
        color = colormap(f)
        axes, fpr, tpr, thresholds, labels = plot_roc_curve(y_true=result[0], y_predict=result[1], label=label, axes=axes, color=color, alpha = 0.7)
        labels_with_auc.extend(labels)

    return axes, labels_with_auc

# combines all variant stratified ROC curves onto one plot, main version
def plot_variant_stratified_roc_curves(axes, records, y_true, y_predict, truth_info_name, annotation_name, tandem_only=False):
    INS = [[],[]]
    DEL = [[],[]]
    INV = [[],[]]
    
    # (results, name of sv type, linestyle for plot)
    sv_types = [(INS, "INS", "-"), (DEL, "DEL", ":"), (INV, "INV", "--")]
    labels_with_auc = ["Variant stratified ROC curves"]

    for r,record in enumerate(records):
        sv_type = record.INFO["SVTYPE"]

        if sv_type == "INS":
            INS[0].append(y_true[r])
            INS[1].append(y_predict[r])
        elif sv_type == "DEL":
            DEL[0].append(y_true[r])
            DEL[1].append(y_predict[r])
        elif sv_type == "INV":
            INV[0].append(y_true[r])
            INV[1].append(y_predict[r])

    for (results, sv_type, line_style) in sv_types:
        if sv_type == "INV" and truth_info_name == "TruvariBench_TP":
            continue # skip INV for Truvari truth
       
        label = truth_info_name + " truth labels and " + annotation_name + " features SV_TYPE: " + sv_type 
        axes, fpr, tpr, thresholds, labels = plot_roc_curve(y_true=results[0], y_predict=results[1], label=label, axes=axes, style=line_style)
        labels_with_auc.extend(labels)

    return axes, labels_with_auc

# alt version of variant stratified plotting plots them separately by different combinations
def plot_variant_stratified_roc_curves_alt(axes, records, y_true, y_predict, truth_info_name, annotation_name, tandem_only=False):
    variant_stratified_results = defaultdict(lambda: [[],[]])
    colormap = pyplot.colormaps['viridis']
    labels_with_auc = ["Variant stratified ROC curves"]

    for r,record in enumerate(records):
        sv_type = record.INFO["SVTYPE"]

        if tandem_only and record.INFO["tr_coverage"] <= 0.9:
            continue 

        variant_stratified_results[sv_type][0].append(y_true[r])
        variant_stratified_results[sv_type][1].append(y_predict[r])
    
    for i,(sv_type,result) in enumerate(sorted(variant_stratified_results.items(), key=lambda x: x[0])):
        if sv_type == "INV" and truth_info_name == "TruvariBench_TP":
            continue # skip INV if using Truvari truth

        label = truth_info_name + " truth labels and " + annotation_name + " features SV_TYPE: " + sv_type

        f = float(i)/float(len(variant_stratified_results))
        color = colormap(f)
        axes, fpr, tpr, thresholds, labels = plot_roc_curve(y_true=result[0], y_predict=result[1], label=label, axes=axes, color=color, alpha = 0.7)
        labels_with_auc.extend(labels)

    return axes, labels_with_auc

# helper function to sort for length and tandem multi stratified, tandem is ranked earlier than non-tandem
def sort_helper_multi_roc(x):
    # example of x: "32 to 64 bp TANDEM" (string)
    if x[0].split(" ")[4] == "TANDEM":
        return (int(x[0].split(" ")[0]), 0)
    else:
        return (int(x[0].split(" ")[0]), 1) 

# stratified ROC by length and tandem
def plot_multi_stratified_roc_curves(axes, records, y_true, y_predict, truth_info_name, annotation_name):
    length_tandem_stratified_results = defaultdict(lambda: [[],[]])
    colormap = pyplot.colormaps['viridis']
    labels_with_auc = ["Length and tandem stratified ROC curves"]

    for r,record in enumerate(records):
        length_bin = int(math.log2(get_length(record) + 1)) 
        bin_start = 2**(length_bin)
        bin_end = 2**(length_bin + 1)

        bin_name = str(bin_start) + " to " + str(bin_end) + " bp"
        is_tandem = record.INFO["tr_coverage"] > 0.9

        if is_tandem:
            length_tandem_stratified_results[bin_name + " TANDEM"][0].append(y_true[r])
            length_tandem_stratified_results[bin_name + " TANDEM"][1].append(y_predict[r])
        else:
            length_tandem_stratified_results[bin_name + " NON-TANDEM"][0].append(y_true[r])
            length_tandem_stratified_results[bin_name + " NON-TANDEM"][1].append(y_predict[r])

    for i,(key,result) in enumerate(sorted(length_tandem_stratified_results.items(), key=sort_helper_multi_roc)):
        label = truth_info_name + " truth labels and " + annotation_name + " features " + key

        f = float(i)/float(len(length_tandem_stratified_results))
        color = colormap(f)
        axes, fpr, tpr, thresholds, labels = plot_roc_curve(y_true=result[0], y_predict=result[1], label=label, axes=axes, color=color, alpha = 0.7)
        labels_with_auc.extend(labels)

    return axes, labels_with_auc

def sort_helper_multi_auc(x):
    # example of x: "TANDEM 32 to 64 bp" (string)
    if x[0].split(" ")[0] == "TANDEM":
        return (0, int(x[0].split(" ")[1]))
    else:
        return (1, int(x[0].split(" ")[1])) 

def plot_multi_stratified_auc_bar(axes, records, y_true, y_predict, truth_info_name, annotation_name):
    length_tandem_stratified_results = defaultdict(lambda: [[],[]])
    colormap = pyplot.colormaps['tab20']

    # iterate through records to assign length bins
    for r,record in enumerate(records):
        length_bin = int(math.log2(get_length(record) + 1))
        bin_start = 2**(length_bin)
        bin_end = 2**(length_bin + 1)

        bin_name = str(bin_start) + " to " + str(bin_end) + " bp"
        is_tandem = record.INFO["tr_coverage"] > 0.9

        if is_tandem:
            length_tandem_stratified_results["TANDEM " + bin_name][0].append(y_true[r])
            length_tandem_stratified_results["TANDEM " + bin_name][1].append(y_predict[r])
        else:
            length_tandem_stratified_results["NON-TANDEM " + bin_name][0].append(y_true[r])
            length_tandem_stratified_results["NON-TANDEM " + bin_name][1].append(y_predict[r])

    num_results = len(length_tandem_stratified_results)
    num_unique_colors = (num_results) // 2

    for i,(key,result) in enumerate(sorted(length_tandem_stratified_results.items(), key=sort_helper_multi_auc)):
        label = key
        color_index = i % num_unique_colors
        f = float(color_index)/float(num_unique_colors)
        
        color = colormap(f)
        axes = plot_auc_bar(y_true=result[0], y_predict=result[1], label=label, axes=axes, color=color, alpha = 0.7)

    return axes


def write_vcf_config(output_dir, train_vcfs, test_vcfs, eval_vcfs, train_contigs, test_contigs, eval_contigs, aucs):
    # maya added txt file with aucs
    with open(os.path.join(output_dir, "train_vcfs.txt"), "w") as f:
        for path in train_vcfs:
            f.write(path + "\n")

    with open(os.path.join(output_dir, "test_vcfs.txt"), "w") as f:
        for path in test_vcfs:
            f.write(path + "\n")

    with open(os.path.join(output_dir, "train_contigs.txt"), "w") as f:
        for contig in train_contigs:
            f.write(contig + "\n")

    with open(os.path.join(output_dir, "test_contigs.txt"), "w") as f:
        for contig in test_contigs:
            f.write(contig + "\n")

    with open(os.path.join(output_dir, "eval_vcfs.txt"), "w") as f:
        for path in eval_vcfs:
            f.write(path + "\n")

    with open(os.path.join(output_dir, "eval_contigs.txt"), "w") as f:
        for contig in eval_contigs:
            f.write(contig + "\n")

    with open(os.path.join(output_dir, "aucs.txt"), "w") as f:
        for auc in aucs:
            f.write(auc + "\n")


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    output_dir = os.path.join("output/", timestamp)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ONT
    # path_to_data = "/Users/mayam/Home/Internships/broad_summer_2024/data/intro/ONT_vcfs/"
    # name_of_file = "_joint_calls_multiannotated_by_single_sample_8x_asm10_20bp.vcf.gz" 

    # PB HiFi
    # path_to_data = "/Users/mayam/Home/Internships/broad_summer_2024/data/intro/PB_HiFi_vcfs/"
    # name_of_file = "_multiannotated_8x_asm10_20bp.vcf.gz"

    # ONT and PB (use this for Union)
    # path_to_data = "/data/mmau/data/intro/ONT_PB_annotated_vcfs/"
    # name_of_file = "_joint_calls_multiannotated_by_single_sample_8x_asm10_20bp.vcf.gz" 

    # PB and VG (use this for short and long reads combined)
    # path_to_data = "/data/mmau/data/intro/with_vg_vcfs/"
    # name_of_file = "_joint_calls_multiannotated_by_single_sample_8x_asm10_20bp.vcf.gz"

    # PB and VG with removing redundant calls using truvari collapse (use this for short and long reads combined)
    # path_to_data = "/data/mmau/data/intro/remove_redundant_calls_truvari/"
    # name_of_file = "_joint_calls_multiannotated_by_single_sample_8x_asm10_20bp.vcf.gz"

    # Truvari collapse VCFs with paragraph (short reads) genotyper
    path_to_data = "/data/mmau/data/intro/paragraph_annotated_vcfs/"
    name_of_file = "_joint_calls_multiannotated_by_single_sample_8x_asm10_20bp.vcf.gz"

    train_vcfs = [
        path_to_data + "HG00621" + name_of_file,
        path_to_data + "HG01928" + name_of_file,
        path_to_data + "HG02572" + name_of_file,
        path_to_data + "HG03098" + name_of_file,
        path_to_data + "HG03492" + name_of_file,
    ]

    test_vcfs = [
        path_to_data + "HG00673" + name_of_file,
        path_to_data + "HG00733" + name_of_file,
    ]

    eval_vcfs = [
        path_to_data + "HG03516" + name_of_file,
    ]

    train_contigs = {"chr1", "chr2", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr21", "chr22", "chrX"}
    test_contigs = {"chr1", "chr2", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr21", "chr22", "chrX"}
    eval_contigs = {"chr1", "chr2", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}

    # Which truth info and annotation names to use (these names are hardcoded in the dataloader to define its behavior)
    truth_info_names = ["Hapestry"]
    truth_info_names = ["Hapestry", "TruvariBench_TP"] 
    annotation_names = ["Hapestry"]
    annotation_names = ["Hapestry", "Sniffles"]

    # Whether to subset the VCFs to >= 50bp
    filter_fn = min50bp
    # filter_fn = None

    # Whether to subset the VCFs to >= 1000bp 
    filter_fn = min1000bp 

    # Initialize some things for plotting
    fig = pyplot.figure()
    fig.set_size_inches(10,8)
    axes = pyplot.axes()

    roc_data = dict()

    # only do hapestry hapestry
    tandem_fig = pyplot.figure()
    tandem_fig.set_size_inches(10,8)
    tandem_axes = pyplot.axes()

    variant_fig = pyplot.figure()
    variant_fig.set_size_inches(10,8)
    variant_axes = pyplot.axes()

    # do all combos
    multi_bar_figs = dict()
    multi_bar_axes = dict()

    length_figs = dict()
    length_axes = dict()

    variant_figs_alt = dict()
    variant_axes_alt = dict() 

    multi_figs = dict()
    multi_axes = dict()

    # number of parallel processes, if running too slowly use n_processes = 1
    n_processes = len(truth_info_names) * len(annotation_names)
    n_processes = 1 

    args = list()

    labels_with_auc = []

    # Set up args for multithreading (each combo of truth/label will get its own thread)
    for truth_info_name in truth_info_names:
        for annotation_name in annotation_names:
            args.append((truth_info_name, annotation_name, train_vcfs, train_contigs, test_vcfs, test_contigs, eval_vcfs, eval_contigs, filter_fn, output_dir))
            
            length_figs[truth_info_name + "_" + annotation_name] = pyplot.figure(figsize=(10,8))
            length_axes[truth_info_name + "_" + annotation_name] = pyplot.axes()

            variant_figs_alt[truth_info_name + "_" + annotation_name] = pyplot.figure(figsize=(10,8))
            variant_axes_alt[truth_info_name + "_" + annotation_name] = pyplot.axes() 

            multi_figs[truth_info_name + "_" + annotation_name] = pyplot.figure(figsize=(10,8))
            multi_axes[truth_info_name + "_" + annotation_name] = pyplot.axes()

            multi_bar_figs[truth_info_name + "_" + annotation_name] = pyplot.figure()
            multi_bar_figs[truth_info_name + "_" + annotation_name].set_size_inches(15,8)
            multi_bar_axes[truth_info_name + "_" + annotation_name] = pyplot.axes()
            multi_bar_axes[truth_info_name + "_" + annotation_name].set_ylim(0.5, 1.0)
            multi_bar_axes[truth_info_name + "_" + annotation_name].set_axisbelow(True) # put grid lines behind bars
            multi_bar_axes[truth_info_name + "_" + annotation_name].grid() # create grid lines


    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.starmap(thread_fn, args)

    # results = [thread_fn(truth_info_names[0], annotation_names[0], train_vcfs, train_contigs, test_vcfs, test_contigs, eval_vcfs, eval_contigs, filter_fn, output_dir)]

    for r,[records, x, feature_indexes, y_true, y_predict, truth_info_name, annotation_name] in enumerate(results):
        label = truth_info_name + " truth labels and " + annotation_name + " features"
        length_axis = length_axes[truth_info_name + "_" + annotation_name]
        variant_axis_alt = variant_axes_alt[truth_info_name + "_" + annotation_name] 
        multi_axis = multi_axes[truth_info_name + "_" + annotation_name]
        multi_bar_axis = multi_bar_axes[truth_info_name + "_" + annotation_name]

        # Plot the individual feature ROCs (without modeling)
        if annotation_name != "Hapestry" and truth_info_name == "TruvariBench_TP":
            cmap = pyplot.get_cmap('tab20')

            for name,index in feature_indexes.items(): # plots dotted lines in all_combos, shows performance from using each of the raw features by thresholding them directly
                # name is DV, DR, GQ, etc.
                if index > 12:
                    y_trivial = x[:,index]
                    y, y_trivial = downsample_test_data(y_true, y_trivial)

                    color = cmap(float(index-12)/(len(feature_indexes)-12))

                    axes, fpr, tpr, thresholds, labels = plot_roc_curve(y_true=y, y_predict=y_trivial, axes=axes, color=color, label=name, style=':')

        tandem_axes, tandem_labels = plot_tandem_stratified_roc_curves(tandem_axes, records, y_true, y_predict, truth_info_name, annotation_name)
        
        length_axis, length_labels = plot_length_stratified_roc_curves(length_axis, records, y_true, y_predict, truth_info_name, annotation_name, tandem_only=True)
        
        variant_axes, variant_labels = plot_variant_stratified_roc_curves(variant_axes, records, y_true, y_predict, truth_info_name, annotation_name)
        variant_axis_alt, variant_labels_alt = plot_variant_stratified_roc_curves_alt(variant_axis_alt, records, y_true, y_predict, truth_info_name, annotation_name)
        
        multi_axis, multi_labels = plot_multi_stratified_roc_curves(multi_axis, records, y_true, y_predict, truth_info_name, annotation_name)
        multi_bar_axis = plot_multi_stratified_auc_bar(multi_bar_axis, records, y_true, y_predict, truth_info_name, annotation_name)
        
        labels_with_auc.extend(tandem_labels + length_labels + variant_labels + multi_labels)

        # write the filtered VCF (BEFORE DOWNSAMPLING)
        write_filtered_vcf(y_predict=y_predict, threshold=0.5, records=records, input_vcf_path=eval_vcfs[0], output_vcf_path=os.path.join(output_dir, label.replace(" ", "_") + ".vcf"))

        y_true, y_predict = downsample_test_data(y_true, y_predict) # downsampling step

        # print shape of y_predict and y_true; print all values in one element
        print("y_predict shape: ", y_predict.shape)
        print("y_true shape: ", y_true.shape)
        print('y_predict y')
        print(y_predict[0], y_true[0])

        axes, fpr, tpr, thresholds, labels = plot_roc_curve(y_true=y_true, y_predict=y_predict, axes=axes, label=label, use_text=True)
        labels_with_auc.append("All combos ROC curves") 
        labels_with_auc.extend(labels) 

        roc_data[label] = (fpr, tpr, thresholds)

    # Generate a legend for axes and force bottom right location
    axes.plot([0, 1], [0, 1], linestyle='--', label="Random classifier", color="gray")
    axes.legend(loc="lower right", fontsize='small')
    fig.savefig(os.path.join(output_dir,"roc_curve_all_combos.png"), dpi=200)

    tandem_axes.plot([0, 1], [0, 1], linestyle='--', label="Random classifier", color="gray")
    tandem_axes.legend(loc="lower right", fontsize='small')
    tandem_fig.savefig(os.path.join(output_dir,"roc_curve_tandem_stratified.png"), dpi=200)

    variant_axes.plot([0, 1], [0, 1], linestyle='--', label="Random classifier", color="gray")
    variant_axes.legend(loc="lower right", fontsize='small')
    variant_fig.savefig(os.path.join(output_dir,"roc_curve_variant_stratified.png"), dpi=200)

    for label,length_axis in length_axes.items():  
        length_axis.plot([0, 1], [0, 1], linestyle='--', label="Random classifier", color="gray")
        length_axis.legend(loc="lower right", fontsize='small')
        length_figs[label].savefig(os.path.join(output_dir,label+"_length.png"), dpi=200)

    for label,variant_axis_alt in variant_axes_alt.items():
        variant_axis_alt.plot([0, 1], [0, 1], linestyle='--', label="Random classifier", color="gray")
        variant_axis_alt.legend(loc="lower right", fontsize='small')
        variant_figs_alt[label].savefig(os.path.join(output_dir,label+"_variant_separated.png"), dpi=200)

    for label,multi_axis in multi_axes.items():
        multi_axis.plot([0, 1], [0, 1], linestyle='--', label="Random classifier", color="gray")
        multi_axis.legend(loc="lower right", fontsize='small')
        multi_figs[label].savefig(os.path.join(output_dir,label+"_multi_length_tandem_separated.png"), dpi=200)

    for label,multi_bar_axis in multi_bar_axes.items():
        x_ticks = multi_bar_axis.get_xticks()
        x_tick_labels = multi_bar_axis.get_xticklabels()

        multi_bar_axis.set_xticks(x_ticks)
        multi_bar_axis.set_xticklabels(x_tick_labels)

        multi_bar_figs[label].tight_layout()
        multi_bar_figs[label].savefig(os.path.join(output_dir,label+"_auc_bar_tandem_length_stratified.png"), dpi=200)
 
    for label,(fpr,tpr,thresholds) in roc_data.items():
        with open(os.path.join(output_dir, label.replace(" ", "_") + "_roc.csv"), "w") as f:
            f.write("threshold,fpr,tpr\n")
            for fpr_val,tpr_val,threshold in zip(fpr,tpr,thresholds):
                f.write(f"{threshold},{fpr_val},{tpr_val}\n")

    # pyplot.show()
    # pyplot.close()

    # Write a bunch of config files for record keeping
    write_vcf_config(output_dir, train_vcfs, test_vcfs, eval_vcfs, train_contigs, test_contigs, eval_contigs, labels_with_auc)


if __name__ == "__main__":
    main()

end = time.time()

print("time of execution is:", (end - start), "s")
