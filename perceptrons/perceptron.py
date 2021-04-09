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
        p = len(self.training_set[0]) - 1
        # print("p:", p)
        w = np.random.uniform(-1, 1, p+1) # array de longitud p+1 con valores random entre -1 y 1
        # print("w:", w)
        error = 1
        self.error_min = p*2

        while error > 0 and i < limit:
            # print("-------------- gen %d --------------" % i)
            if n >= 100 * p:
                # Cada 200 iteraciones tenemos un cambio de w
                w = np.random.uniform(-1, 1, p+1)
                n = 0
            i_x = np.random.randint(0, p+2)
            # print("i_x:", i_x)

            exitation = np.inner(self.training_set[i_x], w)
            # print("exitation:", exitation)
            activ = self.activation(exitation)
            # print("activ:", activ)

            delta_w = (self.learning_rate * (self.expected_output[i_x] - activ)) * self.training_set[i_x]
            # print("########### delta_w terminos ##########")
            # print("learning_rate:", self.learning_rate, "expected_output[i_x] - activ:", self.expected_output[i_x] - activ, "training_set[i_x]:", self.training_set[i_x])
            # print("delta_w:", delta_w)

            w += delta_w
            # print("w_nuevo:", w)

            error = self.error(w)
            # print(error)

            if error < self.error_min:
                self.error_min = error
                self.w_min = w

            i += 1
            n += 1

    # Funcion que recibe array de arrays y con el perceptron entrenado, 
    # devuelve el valor de activacion esperado
    def get_output(self, input):
        print("MIN ERROR:", self.error_min)
        outputs = []
        aux_input = np.array(list(map(lambda t: [1]+t, input)))
        for i in range(len(aux_input)):
            exitation = np.inner(aux_input[i], self.w_min)
            outputs.append(self.activation(exitation))
        return outputs

    @abstractmethod
    def activation(self, exitation):
        pass

    # funcion que calcula el error en cada iteracion utilizando el conjunto de entrenamiento,
    # la salida esperada, el vector de pesos y la longitud del conjunto de entranamiento
    @abstractmethod
    def error(self,w):
        pass
    
