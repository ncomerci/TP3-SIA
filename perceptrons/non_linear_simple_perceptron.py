from perceptrons.perceptron import Perceptron
import numpy as np
import math

class NonLinearSimplePerceptron(Perceptron):

    def __init__(self, training_set, expected_output,learning_rate):
        super().__init__(training_set,expected_output,learning_rate)
    
    def activation(self, excited_state):
        return math.tanh(excited_state)
         
    def error(self,w):