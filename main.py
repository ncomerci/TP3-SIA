from perceptrons.simple_perceptron import SimplePerceptron
from perceptrons.linear_simple_perceptron import LinearSimplePerceptron
from perceptrons.non_linear_simple_perceptron import NonLinearSimplePerceptron
from perceptrons.multilayer_perceptron import MultilayerPerceptron

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
adaptive_eta = config["multilayer_perceptron"]["adaptive_eta"]
batch = config["multilayer_perceptron"]["batch"]
momentum = config["multilayer_perceptron"]["momentum"]
rows_per_entry = config["training_file_lines_per_entry"]

read_training_txt = import_and_parse_data(training_file, rows_per_entry)
read_output_txt = import_and_parse_data(output_file, 1)
# print(read_training_txt)
# print(read_output_txt)
total_input = len(read_training_txt)
limit = math.ceil(total_input * training_amount)

training_set = read_training_txt[:limit]
generalize_set = read_training_txt[limit:]

learn_expected = read_output_txt[:limit]
generalize_expected = read_output_txt[limit:]

# amount = 0.1 
# while amount < 1:
#     limit = math.ceil(total_input * amount)
    
#     training_set = read_training_tsv[:limit]
#     generalize_set = read_training_tsv[limit:]
#     learn_expected = read_output_tsv[:limit]
#     generalize_expected = read_output_tsv[limit:]

# #    print("============== TRAINING %f  ===============" %amount)
#     e = []
#     for j in range(3):
#         sp = perceptrons[perceptron](training_set, learn_expected, learning_rate)
#         sp.train(max_iterations)
#         out = sp.get_output(generalize_set)
#         error = 0
#         for real_output, expected_output in zip(out, generalize_expected):  
#             #print(f"OUTPUT: {real_output}, EXPECTED: {expected_output}, ERROR: {abs(real_output - expected_output)}")
         
#             error += abs(real_output - expected_output)
            
#             # QUE TAN BIEN APRENDE  --> esto es con train y pasandole en get_output todo el train 
#             # QUE TAN BIEN GENERALIZA --> esto es con cte*train y  1-cte * generalized
#         e.append(error/len(out)) 
#     print(sum(e)/len(e)) 
#     amount += 0.05 

if (perceptron == "multilayer_perceptron"): 
    sp = MultilayerPerceptron(training_set, learn_expected, learning_rate, hidden_layers, adaptive_eta, batch, momentum)
else: 
    sp = perceptrons[perceptron](training_set, learn_expected, learning_rate)
    
sp.train(epochs_amount)             # Train perceptron with a part of the dataset 
out = sp.get_output(training_set)   # Get real output based on the weights obtained in the training 
print(out) 
