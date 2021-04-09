# uses the "step" activaction function in order to learn the "AND" problem 

import numpy as np
from perceptrons.simple_perceptron import simple_perceptron 

training_set = [ [1,1], [1,-1], [-1,-1], [1,1]]
expected_output = [-1, -1, -1, 1]
  
learning_rate  = 0.01

simple_perceptron(1, 0.1, learning_rate, training_set, expected_output)