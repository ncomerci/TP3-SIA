from perceptrons.simple_perceptron import SimplePerceptron

training = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
and_output = [-1, -1, -1, 1]
xor_output = [1, 1, -1, -1]

sp = SimplePerceptron(training, and_output, 0.01)

sp.train(100000)

print("ACTIVATION:", sp.get_output([[1, 1], [1, -1], [-1, -1], [1, 1]]))