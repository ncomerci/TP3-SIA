from perceptrons.perceptron import Perceptron
import numpy as np

class LinearSimplePerceptron(Perceptron):

    def __init__(self, training_set, expected_output,learning_rate):
        super().__init__(training_set,expected_output,learning_rate)
    
    def activation(self, excited_state):
        return excited_state
         
    def error(self,w):
        error = 0
        for i in range(len(self.training_set)):
            activation_state = np.inner(self.training_set[i], w) # internal product: sum (e[i_x]*w_i) --> hiperplano
            # error += (self.expected_output[i] - activation_state)**2
            error += abs(self.expected_output[i] - activation_state)
        # return 0.5 * error
        return error

    def delta_w_correction(self,h):
        return 1