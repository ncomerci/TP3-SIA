# p is the size of the training set 
# w is an array of synaptic weights
# N is the dimension of the Input --> as we are working with points N=2
# x is an array with the training_set (Input) It already has an E0=1 so we do not add w0
# y is an array of expected outputs

# Desconozco los pesos sinápticos que tiene que tener la neurona. Dsp de varias iteraciones voy a obtener el w que necesito. 
# La salida de la neurona es el estado de activación. Pero yo quiero que la salida sea Z (expected_output)
# Si la salidad de la neurona (O, activation_state) es distinta a lo que yo queria (Z) --> le aplico la correcion delta_w, sino la dejo como estaba, esa corrección depende de mi entrada (E)
# El objetivo es que el perceptron converja a la solución.

from abc import ABC, abstractmethod
import math
import random
import numpy as np

class Perceptron(ABC):

    def __init__(self, training_set, expected_output,learning_rate):
        self.training_set = np.array(list(map(lambda t: [1]+t, training_set)))
        self.expected_output = np.array(expected_output)
        self.learning_rate = learning_rate
        self.w_min = None
        self.error_min = None
    
    def train(self, limit):
        i = 0
        n = 0
        p = len(self.training_set)
        dimension = len(self.training_set[0])   
        w = np.random.uniform(-1, 1, dimension) # array de longitud p+1 con valores random entre -1 y 1  
    
        error = 1
        # self.error_min = p*2
        self.error_min = float('inf')
        while error > 0 and i < limit:
            
            if n >= 100 * p: # initialize weights again
                w = np.random.uniform(-1, 1,dimension)  
                n = 0
                
            i_x = np.random.randint(0, p) # get a random point index from the training set  
         
            excited_state = np.inner(self.training_set[i_x], w) # internal product: sum (e[i_x]*w_i) --> hiperplano
 
            activation_state = self.activation(excited_state) 
 
            delta_w = (self.learning_rate * (self.expected_output[i_x] - activation_state)) * self.training_set[i_x] * self.delta_w_correction(excited_state)

            w += delta_w 
            
            error = self.error(w)

            if error < self.error_min:
                self.error_min = error
                self.w_min = w

            i += 1
            n += 1

    # Funcion que recibe array de arrays y con el perceptron entrenado, 
    # devuelve el valor de activation_state sea el esperado
    def get_output(self, input):
        print("MIN ERROR:", self.error_min)
        outputs = []
        aux_input = np.array(list(map(lambda t: [1]+t, input)))
        for i in range(len(aux_input)):
            excited_state = np.inner(aux_input[i], self.w_min)
            outputs.append(self.activation(excited_state))
        return outputs

    @abstractmethod
    def activation(self, excited_state):
        pass

    # funcion que calcula el error en cada iteracion utilizando el conjunto de entrenamiento,
    # la salida esperada, el vector de pesos y la longitud del conjunto de entranamiento
    @abstractmethod
    def error(self,w):
        pass
    
    # en el perceptron no lineal hay que multiplicar delta_w * g'(h)
    def delta_w_correction(self,h):
        return 1

#      w0 e1 e2     
#     [ 1 -1,1    1 1,-1    1 -1,-1    1 1,1  ]  training set (E)
# and   -1         -1          -1         1      expected_outputs (Z) --> es lo que quiero aprender
# xor    1          1          -1        -1
