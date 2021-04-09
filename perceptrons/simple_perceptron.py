# p is the size of the training set 
# w is an array of synaptic weights
# N is the dimension of the Input --> as we are working with points N=2
# x is an array with the training_set (Input) It already has an E0=1 so we do not add w0
# y is an array of expected outputs

# Desconozco los pesos sinápticos que tiene que tener la neurona. Dsp de varias iteraciones voy a obtener el w que necesito. 
# La salida de la neurona es el estado de activación. Pero yo quiero que la salida sea Z (expected_output)
# Si la salidad de la neurona (O, activation_state) es distinta a lo que yo queria (Z) --> le aplico la correcion delta_w, sino la dejo como estaba, esa corrección depende de mi entrada (E)
# El objetivo es que el perceptron converja a la solución.

from perceptrons.perceptron import Perceptron
import numpy as np

class SimplePerceptron(Perceptron):

    def __init__(self, training_set, expected_output,learning_rate):
        super().__init__(training_set,expected_output,learning_rate)
    
    def activation(self, excited_state):
        return np.sign(excited_state)

    def error(self,w):
        # Para cada elemento del conjunto de entrada aplicandole su peso correspondiente, 
        # tengo que ver si da la salida esperada, en caso de que no de voy acumulando dicho error
        training_size = len(self.training_set)
        error = 0
        for i in range(training_size):
            excited_state = np.inner(self.training_set[i], w)
            if abs(self.activation(excited_state) - self.expected_output[i]) != 0:
                print("Error for line %d" % i)
            error += abs(self.activation(excited_state) - self.expected_output[i])
        return error
