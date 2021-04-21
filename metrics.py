from perceptrons.linear_simple_perceptron import LinearSimplePerceptron
from perceptrons.non_linear_simple_perceptron import NonLinearSimplePerceptron

import random
import numpy as np 

class Metrics: 
    
    __init__(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix
    
    def accuracy: 
    
    
    
    def cross_validation(self, k_fold, data_set, expected_output, learning_rate, amount, perceptron_type):
        
        if not (len(data_set) % k_fold == 0) : 
            print("Choose another partition size")
            exit()
            
        all_indexes = list(range(len(data_set)))
        random.shuffle(indexes)
        splitted_indexes = np.array_split(np.array(all_indexes), k_fold)
  
        #  | test  |        |   |   |   | 
        #  |       | test   |   |   |   | 

        if perceptron_type == LINEAR_SP: 
            sp_class = LinearSimplePerceptron
        elif perceptron_type == NON_LINEAR_SP: 
            sp_class = NonLinearSimplePerceptron
        else: 
            print("Error")
            
        best = None
        for indexes in splitted_indexes: 
            compl_indexes = set(all_indexes) - set(indexes.tolist())
            
            testing_set = [data_set[i] for i in indexes]
            training_set = [data_set[i] for i in compl_indexes]
            sub_expected_output = [data_set[i] for i in compl_indexes] 
            
            sp = sp_class(training_set, sub_expected_output, learning_rate)   
            
            sp.train(amount)
            real_output = sp.get_output(testing_set) 
            
            self.calculate_confusion_matrix([1, -1], real_output, expected_output)
            # calcular la confusion matrix con ese output 
            # calculo metrics 
            
            # guardo best = testing  --> la mejor config training/test tal que los pesos sean los mejores, el error sea minimo, la accuracy sea mayor :D
            
    
    # Confusion matrix --> en las columnas los valores predecidos y en las filas el real
    def calculate_confusion_matrix(self, classes, real_output, expected_output):
        
        # true positive  --> par y la red dice que es par
        # true negative  --> impar y la red dice que es impar
        # false positive --> impar y la red me dice par
        # false negative --> par y la red dice que es impar 
        
        # imagenes     par  impar 
        # par           5     0
        # impar         0     5

        # xor         1     -1 
        #  1          2     0
        # -1          0     2
        
        bias = 0.01
        
        if( real - expected_output < bias ) # true positive 
            if expe -1

        if  
        
            
         
           
        
        
        
        
    

 