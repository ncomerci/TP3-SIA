from perceptrons.simple_perceptron import SimplePerceptron
from perceptrons.linear_simple_perceptron import LinearSimplePerceptron
from perceptrons.non_linear_simple_perceptron import NonLinearSimplePerceptron
import csv, json

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


read_training_tsv = csv.reader(training_file, delimiter="\t")
read_output_tsv = csv.reader(output_file, delimiter="\t")

training_set = []
for row in read_training_tsv:
    training_set.append(list(map(lambda x: float(x), row)))
    
expected_output = []
for row in read_output_tsv:
    expected_output.append(list(map(lambda x: float(x), row)))

sp = perceptrons[perceptron](training_set, expected_output, learning_rate)

sp.train(max_iterations)

print("ACTIVATION:", sp.get_output(training_set))