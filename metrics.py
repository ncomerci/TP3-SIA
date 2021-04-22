from perceptrons.linear_simple_perceptron import LinearSimplePerceptron
from perceptrons.non_linear_simple_perceptron import NonLinearSimplePerceptron
from perceptrons.multilayer_perceptron import MultilayerPerceptron
from constants import * 

import random
import numpy as np 

class Metrics: 
     
    # TP FN 
    # FP TN
    def accuracy(self,confusion_matrix): 
        # TP + TN / TODOS 
        
        true = confusion_matrix[0][0] + confusion_matrix[1][1]
        false = confusion_matrix[0][1] + confusion_matrix[1][0]
        
        return true / (false + true) 
        
    def precision(self,confusion_matrix): 
        # TP   /  TP+FP
        
        true = confusion_matrix[0][0] + confusion_matrix[1][1]
        false = confusion_matrix[1][0]
        
        return true / false           
    
    def recall(self,confusion_matrix): 
        # TP   /  TP+FN
        
        true = confusion_matrix[0][0] + confusion_matrix[1][1]
        false = confusion_matrix[0][1]
        
        return true / false 
    
    def f1_score(self,confusion_matrix): 
        
        precision = self.precision(confusion_matrix)
        recall = self.recall(confusion_matrix)
        
        return (2*precision*recall) / (precision + recall) 
         
   
    def cross_validation(self,k_fold, data_set, expected_output, learning_rate, amount, perceptron_type,hidden_layers):
        
        if not (len(data_set) % k_fold == 0) : 
            print("Choose another partition size")
            exit()
            
        all_indexes = list(range(len(data_set)))
        random.shuffle(all_indexes)
        splitted_indexes = np.array_split(np.array(all_indexes), k_fold)
        print()
        #  | test  |        |   |   |   | 
        #  |       | test   |   |   |   | 

        if perceptron_type == LINEAR_SP: 
            sp_class = LinearSimplePerceptron
        elif perceptron_type == NON_LINEAR_SP: 
            sp_class = NonLinearSimplePerceptron
        elif perceptron_type == MULTILAYER:
            sp_class = MultilayerPerceptron
            
        best_metric = float('inf')
        best_indexes = None
        neuron = None
        for indexes in splitted_indexes: 
            compl_indexes = set(all_indexes) - set(indexes.tolist())
            print(compl_indexes)
            
            testing_set = [data_set[i] for i in indexes]
            test_expected_output = [expected_output[i] for i in indexes]
            training_set = [data_set[i] for i in compl_indexes]
            sub_expected_output = [expected_output[i] for i in compl_indexes]
            if(perceptron_type == MULTILAYER):
                adaptive_eta_params = [False, 0, 0, 0]
                sp = sp_class(training_set, sub_expected_output, learning_rate, hidden_layers, adaptive_eta_params)
            else:
                sp = sp_class(training_set, sub_expected_output, learning_rate)   
            
            sp.train(amount)
            real_output = sp.get_output(testing_set) 
            
            confusion_matrix = self.calculate_confusion_matrix([1, -1], real_output, test_expected_output)
            print(confusion_matrix)
            metric = self.accuracy(confusion_matrix) 
            
            if(metric < best_metric): 
                best_metric = metric 
                neuron = sp
                best_indexes = indexes
        print(best_metric)
        print(best_indexes)
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
        print(real_output)
        values = [elem[0] for elem in real_output]
        print(expected_output)
        bias = 0.1
        matrix = [[0,0], [0,0]]
        for i in range(len(real_output)): 
            
            if(values[i] - expected_output[i] <= bias ): # TN o TP 
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

        
            
         
           
        
        
        
        
    

 