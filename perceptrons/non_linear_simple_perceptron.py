from perceptrons.perceptron import Perceptron
from normalize import Normalization
import numpy as np
import math

class NonLinearSimplePerceptron(Perceptron):

    def __init__(self, training_set, expected_output,learning_rate):
        self.norm = Normalization(training_set,expected_output, (-1,1))
        super().__init__( self.norm.input_normalized(training_set), self.norm.expected_normalized(),learning_rate) #As we are using tanh we need to normalize
    
    def activation(self, excited_state):
        return np.tanh(excited_state)
         
    def error(self,w):
        error = 0
        for i in range(len(self.training_set)):
            excited_state = np.inner(self.training_set[i], w) # internal product: sum (e[i_x]*w_i) --> hiperplano
            activation_state = self.activation(excited_state)
            error += (self.expected_output[i] - activation_state)**2
        return 0.5 * error
        
    def delta_w_correction(self,h):
        return (1 - math.tanh(h)**2)

    def get_output(self, input):
        norm_input =  self.norm.input_normalized(input)
        norm_output = super().get_output(norm_input)
        return  self.norm.unnormalize(norm_output)

