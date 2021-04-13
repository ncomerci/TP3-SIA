from perceptrons.simple_perceptron import SimplePerceptron
from perceptrons.linear_simple_perceptron import LinearSimplePerceptron
from perceptrons.non_linear_simple_perceptron import NonLinearSimplePerceptron

import csv, json
import math

import csv
import itertools


with open("config.json") as f:
    config = json.load(f)

training_file = open(config["training_tsv_path"])
output_file = open(config["output_tsv_path"])
learning_rate = config["learning_rate"]
perceptrons = {
    "step_simple_perceptron": SimplePerceptron,
    "linear_simple_perceptron": LinearSimplePerceptron,
    "non_linear_simple_perceptron": NonLinearSimplePerceptron
}
perceptron = config["perceptron_type"]
max_iterations = config["max_iterations"]
training_amount = config["training_amount"]

read_training_tsv = list(csv.reader(training_file, delimiter="\t"))
read_training_tsv = list(map(lambda array: list(map(lambda x: float(x), array)), read_training_tsv))
read_output_tsv = list(csv.reader(output_file, delimiter="\t"))
read_output_tsv = list(map(lambda elem: float(elem[0]), read_output_tsv))
total_input = len(read_training_tsv)
limit = math.ceil(total_input * training_amount)

training_set = read_training_tsv[:limit]
generalize_set = read_training_tsv[limit:]

learn_expected = read_output_tsv[:limit]
generalize_expected = read_output_tsv[limit:]

sp = perceptrons[perceptron]( training_set, learn_expected, learning_rate) 

sp.train(max_iterations)
out = sp.get_output(training_set)

print(out)
