import numpy as np

# Normalize data between -1 and 1
def normalize(dataset): 
    max_val = max(dataset)
    min_val = min(dataset)
    return list(map(lambda elem: 2*(elem - min_val)/(max_val - min_val) - 1, dataset))
    # return np.array(list(map(lambda elem: 2*(elem - min_val)/(max_val - min_val) - 1, dataset)))
    # return [2*(elem - min_val)/(max_val - min_val) - 1 for elem in data]

def normalize_training_set(tr_set):
    np_array = np.array(tr_set)
    norm_ts_1 = normalize(np_array[:,0]) #get first col
    # print("ts_1:", norm_ts_1)
    norm_ts_2 = normalize(np_array[:,1])
    # print("ts_2:", norm_ts_2)
    norm_ts_3 = normalize(np_array[:,2])
    # print("ts_3:", norm_ts_3)   
    norm_training_set = []
    
    for (e1,e2,e3) in zip(norm_ts_1,norm_ts_2,norm_ts_3):
        norm_training_set.append([e1,e2,e3])
    return norm_training_set
    # ret = np.concatenate((norm_ts_1, norm_ts_2, norm_ts_3))
    # print("result:", ret)
    # return ret
                                                            