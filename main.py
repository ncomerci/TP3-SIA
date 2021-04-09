from perceptrons.simple_perceptron import SimplePerceptron
from perceptrons.linear_simple_perceptron import LinearSimplePerceptron
from perceptrons.non_linear_simple_perceptron import NonLinearSimplePerceptron
import csv

training = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
and_output = [-1, -1, -1, 1]
xor_output = [1, 1, -1, -1]
or_output = [1, 1, -1, 1]

training_file = open("training_set2.tsv")
output_file = open("expected_output2.tsv")
read_tsv = csv.reader(training_file, delimiter="\t")
read_tsv2 = csv.reader(output_file, delimiter="\t")

training_set = []
for row in read_tsv:
    training_set.append(list(map(lambda x: float(x), row)))
expected = []
for row in read_tsv2:
    expected.append(list(map(lambda x: float(x), row)))
#sp = LinearSimplePerceptron(training_set, expected, 0.01)
sp = NonLinearSimplePerceptron(training_set, expected, 0.01)

sp.train(10000)

print("ACTIVATION:", sp.get_output([[4.4793,-4.0765,4.4558]]))