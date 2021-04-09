from perceptrons.perceptron import Perceptron
import numpy as np

class SimplePerceptron(Perceptron):

    def __init__(self, training_set, expected_output,learning_rate):
        super().__init__(training_set,expected_output,learning_rate)
    
    def activation(self, exitation):
        return np.sign(exitation)

    def error(self,w):
        #Para cada elemento del conjunto de entrada aplicandole su peso correspondiente, 
        # tengo que ver si da la salida esperada, en caso de que no de voy acumulando dicho error
        training_size = len(self.training_set)
        error = 0
        for i in range(training_size):
            excited_state = np.inner(self.training_set[i], w)
            # if abs(self.activation(excited_state) - self.expected_output[i]) != 0:
            #     print("Error for line %d" % i)
            error += abs(self.activation(excited_state) - self.expected_output[i])
        return error
