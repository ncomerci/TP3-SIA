import numpy as np

class Normalization:

    def __init__(self, training_set, expected_output, extreme_values):
        self.training_set = np.array(training_set)
        self.expected_output = expected_output
        self.min_val_training = [min(self.training_set[:,0]), min(self.training_set[:,1]), min(self.training_set[:,2])]
        self.max_val_training = [max(self.training_set[:,0]), max(self.training_set[:,1]), max(self.training_set[:,2])]
        self.min_val_expected = min(expected_output)
        self.max_val_expected = max(expected_output)

    # Normalize data between -1 and 1

    def expected_normalized(self):
        return self.normalize(self.expected_output, self.max_val_expected, self.min_val_expected)

    def normalize(self,dataset,max, min): 
        return list(map(lambda elem: 2*(elem - min)/(max - min) - 1, dataset)) 

    def input_normalized(self,input):
        np_array = np.array(input)
  
        norm_ts_1 = self.normalize(np_array[:,0],self.max_val_training[0],self.min_val_training[0]) #get first col 
        norm_ts_2 = self.normalize(np_array[:,1],self.max_val_training[1],self.min_val_training[1]) 
        norm_ts_3 = self.normalize(np_array[:,2],self.max_val_training[2],self.min_val_training[2])
        
        norm_training_set = []
        
        for (e1,e2,e3) in zip(norm_ts_1,norm_ts_2,norm_ts_3):
            norm_training_set.append([e1,e2,e3])
            
        return norm_training_set
                                  
    def unnormalize(self, dataset):
        return list(map(lambda elem: (elem + 1) / 2 * (self.max_val_expected - self.min_val_expected) - self.min_val_expected, dataset)) 
    