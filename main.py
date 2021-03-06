from perceptrons.simple_perceptron import SimplePerceptron
from perceptrons.linear_simple_perceptron import LinearSimplePerceptron
from perceptrons.non_linear_simple_perceptron import NonLinearSimplePerceptron
from perceptrons.multilayer_perceptron import MultilayerPerceptron
from metrics import Metrics
from constants import * 

import csv, json
import math

import csv
import itertools

def import_and_parse_data(file, rows_per_entry):
    datafile = open(file, 'r')
    datareader = csv.reader(datafile, delimiter=' ')
    data = []
    row_count = 0
    entry = []
    for row in datareader:
        if row_count < rows_per_entry:
            entry += [float(a) for a in row if a != '']
        else:
            row_count = 0
            if len(entry) == 1:
                data.append(entry[0]) 
            else:
                data.append(entry) 
            entry = [float(a) for a in row if a != '']
        row_count += 1

    if len(entry) == 1:
        data.append(entry[0]) 
    else:
        data.append(entry) 

    return data

with open("config.json") as f:
    config = json.load(f)

training_file = config["training_file_path"]
output_file = config["output_file_path"]
learning_rate = config["learning_rate"]
perceptrons = {
    "step_simple_perceptron": SimplePerceptron,
    "linear_simple_perceptron": LinearSimplePerceptron,
    "non_linear_simple_perceptron": NonLinearSimplePerceptron, 
    "multilayer_perceptron": MultilayerPerceptron
}
perceptron = config["perceptron_type"]
max_iterations = config["max_iterations"]
training_amount = config["training_amount"] 
hidden_layers = config["multilayer_perceptron"]["hidden_layers"]
epochs_amount = config["multilayer_perceptron"]["epochs_amount"]
adaptive_eta = config["multilayer_perceptron"]["adaptive_eta"]["use"]
adaptive_eta_increase = config["multilayer_perceptron"]["adaptive_eta"]["increase_by"]
adaptive_eta_decrease = config["multilayer_perceptron"]["adaptive_eta"]["decrease_by"]
adaptive_eta_max_iterations = config["multilayer_perceptron"]["adaptive_eta"]["max_iterations"]
batch = config["multilayer_perceptron"]["batch"]
momentum = config["multilayer_perceptron"]["momentum"]
rows_per_entry = config["training_file_lines_per_entry"]

read_training_txt = import_and_parse_data(training_file, rows_per_entry)
read_output_txt = import_and_parse_data(output_file, 1)
 
total_input = len(read_training_txt)
limit = math.ceil(total_input * training_amount)

training_set = read_training_txt[:limit]
generalize_set = read_training_txt[limit:]

learn_expected = read_output_txt[:limit]
generalize_expected = read_output_txt[limit:]


if (perceptron == MULTILAYER): 
    adaptive_eta_params = [adaptive_eta, adaptive_eta_increase, adaptive_eta_decrease, adaptive_eta_max_iterations]
    sp = MultilayerPerceptron(training_set, learn_expected, learning_rate, hidden_layers, adaptive_eta_params,batch=batch,momentum=momentum)
    amount = epochs_amount
else: 
    sp = perceptrons[perceptron](training_set, learn_expected, learning_rate)
    amount = max_iterations

#if(cross_validation):                                                           # Choose best training set/testing set partition
#    Metrics().cross_validation(2, read_training_txt , read_output_txt, learning_rate, amount, perceptron, hidden_layers)   
#else:   
sp.train(amount)                                                            # Train perceptron with a part of the dataset 
out = sp.get_output(generalize_set if generalize_set else training_set)     # Get real output based on the weights obtained in the training 
print("OUTPUT:")
print(out)
error = 0
for real_output, expected  in zip(out, read_output_txt):  
    
    if (perceptron == MULTILAYER):  
        error += abs(real_output[0] - expected)
    else: 
        error += abs(real_output  - expected ) 
print(f"ABSOLUTE ERROR: {error/len(out)}")
