from perceptrons.linear_simple_perceptron import LinearSimplePerceptron
from perceptrons.non_linear_simple_perceptron import NonLinearSimplePerceptron
from constants import * 

import random
import numpy as np 

class Metrics: 
     
    # TP FN 
    # FP TN
    def accuracy(confusion_matrix): 
        # TP + TN / TODOS 
        
        true = confusion_matrix[0][0] + confusion_matrix[1][1]
        false = confusion_matrix[0][1] + confusion_matrix[1][0]
        
        return true / (false + true) 
        
    def precision(confusion_matrix): 
        # TP   /  TP+FP
        
        true = confusion_matrix[0][0] + confusion_matrix[1][1]
        false = confusion_matrix[1][0]
        
        return true / false           
    
    def recall(confusion_matrix): 
        # TP   /  TP+FN
        
        true = confusion_matrix[0][0] + confusion_matrix[1][1]
        false = confusion_matrix[0][1]
        
        return true / false 
    
    def f1_score(confusion_matrix): 
        
        precision = self.precision(confusion_matrix)
        recall = self.recall(confusion_matrix)
        
        return (2*precision*recall) / (precision + recall) 
         
   
    def cross_validation( k_fold, data_set, expected_output, learning_rate, amount, perceptron_type):
        
        if not (len(data_set) % k_fold == 0) : 
            print("Choose another partition size")
            exit()
            
        all_indexes = list(range(len(data_set)))
        random.shuffle(all_indexes)
        splitted_indexes = np.array_split(np.array(all_indexes), k_fold)
  
        #  | test  |        |   |   |   | 
        #  |       | test   |   |   |   | 

        if perceptron_type == LINEAR_SP: 
            sp_class = LinearSimplePerceptron
        elif perceptron_type == NON_LINEAR_SP: 
            sp_class = NonLinearSimplePerceptron
        else: 
            print("Error")
            
        best_metric = float('inf')
        best_indexes = None
        neuron = None
        for indexes in splitted_indexes: 
            compl_indexes = set(all_indexes) - set(indexes.tolist())
            
            testing_set = [data_set[i] for i in indexes]
            training_set = [data_set[i] for i in compl_indexes]
            sub_expected_output = [data_set[i] for i in compl_indexes] 
   
            sp = sp_class(training_set, sub_expected_output, learning_rate)   
            
            sp.train(amount)
            real_output = sp.get_output(testing_set) 
            
            confusion_matrix = self.calculate_confusion_matrix([1, -1], real_output, expected_output)
             
            metric = self.accuracy(confusion_matrix) 
            
            if(metric < best_metric): 
                best_metric = metric 
                neuron = sp
                best_indexes = indexes

        return neuron, best_metric, best_indexes
    
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
        matrix = [[0,0], [0,0]]
        
        for i in range(len(real_output)): 
            
            if(real_output[i] - expected_output[i] <= bias ): # TN o TP 
                if expected_output[i] == 1:
                    matrix[0][0] += 1                 
                elif expected_output[i] == -1: 
                    matrix[1][1] += 1
                else: 
                    print("error")
            
            else: #FN o FP
                if expected_output[i] == 1: #FN 
                    matrix[0][1] += 1
                elif expected_output[i] == -1: #FP 
                    matrix[1][0] += 1 
                else: 
                    print("error")
        
        return matrix

        
            
         
           
        
        
        
        
    

 