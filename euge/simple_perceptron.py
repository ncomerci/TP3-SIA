# p is the size of the training set 
# w is an array of synaptic weights
# N is the dimension of the Input --> as we are working with points N=2
# x is an array with the training_set (Input) It already has an E0=1 so we do not add w0
# y is an array of expected outputs

# Desconozco los pesos sinápticos que tiene que tener la neurona. Dsp de varias iteraciones voy a obtener el w que necesito. 
# La salida de la neurona es el estado de activación. Pero yo quiero que la salida sea Z (expected_output)
# Si la salidad de la neurona (O, activation_state) es distinta a lo que yo queria (Z) --> le aplico la correcion delta_w, sino la dejo como estaba, esa corrección depende de mi entrada (E)
# El objetivo es que el perceptron converja a la solución.

import math
import random
import numpy as np

N=2
MAX_ITERATIONS = 100
def simple_perceptron(activation_function, error_function, learning_rate, training_set, expected_outputs):
    
    #agregar un 1 al PRINCIPIO de cada lista del training set 
    
    weights = []
    for i in range(N+1): # initialize weights with random values
        weights.append(random.randint(1,N+1) * 2 - 1)
        
    p = len(training_set) 
    error = 1 
    error_min = p*2
    
    i=0
    n=0
    while error > 0 and i < MAX_ITERATIONS: 
        
        if n > 100 * p :  # initialize weights again
            for i in range(N+1):
                weights[i] = random.randint(1,N+1) * 2 - 1
            n = 0
            
        i_x = random.randint(1,p)  # get a random point index from the training set
        
        excited_state = 0
        for coordinate in range(N+1):
            print(coordinate)
            excited_state +=  training_set[i_x][coordinate] * weights[coordinate] # internal product: sum (e[i_x]*w_i) --> hiperplano
        
        activation_state = np.sign(excited_state)   #activation_function(excited_state)
        
        for i in range(N+1):
            delta_w += learning_rate * float(expected_outputs[i_x] - activation_state) * float(training_set[i_x][i])  # ???? calculate correction 
            weights[i] += delta_w #list(map(lambda w: w + delta_w, weights))  # new_weight = old_weight + delta_w
        
        error -= 0.1 #error_function(training_set, expected_outputs, weights)
        if error < error_min: 
            error_min = error 
            w_min = weights 
   
        i+=1
        n+=1
    
    print(activation_state)
    
   #      w0 e1 e2     
   #     [ 1 -1,1    1 1,-1    1 -1,-1    1 1,1  ]  training set (E)
   # and   -1         -1          -1         1      expected_outputs (Z) --> es lo que quiero aprender
   # xor    1          1          -1        -1
    
       
     