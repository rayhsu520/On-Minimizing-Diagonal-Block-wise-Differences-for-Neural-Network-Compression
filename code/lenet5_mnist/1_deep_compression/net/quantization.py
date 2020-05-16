'''
Modified from https://github.com/mightydeveloper/Deep-Compression-PyTorch
'''
import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix
import util
import scipy.linalg



def apply_weight_sharing(model, model_mode, bits_dict):
    """
    Applies weight sharing to the given model
    """
    old_weight_list, new_weight_list, quantized_index_list, quantized_center_list = sequential_weight_sharing(model.children(), model_mode, bits_dict)
    
    return old_weight_list, new_weight_list, quantized_index_list, quantized_center_list

def sequential_weight_sharing(m, model_mode, bits_dict):

    old_weight_list = []
    new_weight_list = []
    quantized_index_list = []
    quantized_center_list = []

    for module in m:
        dev = module.weight.device
        old_weight = module.weight.data.cpu().numpy()
        shape = old_weight.shape
        
        # convolution layer
        if len(shape)>2:
            bits = bits_dict['conv']
            if model_mode == 'd':
                #skip convolution layer
                continue  
            weight = old_weight.reshape(1, -1)

        elif len(shape) == 1: 
            continue
        elif len(shape) == 2:
            bits = bits_dict['fc']
            weight = old_weight.reshape(1, -1)

        print(module)
        zero_index = np.where(weight==0)[1]
        mat = weight[0]
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num=2**int(bits))
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(mat.reshape(-1,1))
        cluster_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        cluster_weight[zero_index]=0.0
        new_weight = cluster_weight
        new_weight = new_weight.reshape(shape)
        module.weight.data = torch.from_numpy(new_weight).to(dev)
        quantized_index = kmeans.labels_.reshape(shape)
        quantized_center = kmeans.cluster_centers_

        old_weight_list.append(old_weight)
        new_weight_list.append(new_weight)
        quantized_index_list.append(quantized_index)
        quantized_center_list.append(quantized_center)

    return old_weight_list, new_weight_list, quantized_index_list, quantized_center_list
'''
def sequential_weight_sharing(m, bits=4):

    old_weight_list = []
    new_weight_list = []
    quantized_index_list = []
    quantized_center_list = []
    #print(m.
    #for module in m:
    for module in m:
        #print(module.grad.data.cpu().numpy())
        #print(module.name)
        dev = module.weight.device
        old_weight = module.weight.data.cpu().numpy()
        shape = old_weight.shape
        
        # convlution layer and bn layer 
        if len(shape)>2 or len(shape)==1:
            #skip convlution layer and bn layer
            continue  
            weight = old_weight.reshape(1, -1)
            if shape[1] < 2**bits:
                old_weight_list.append(old_weight)
                new_weight_list.append(old_weight)
                continue
        elif len(shape)==2:
            weight = old_weight.reshape(1, -1)
		zero_index = np.where(weight==0)[1]
        print(module)
        for i in range(len(weight)):
            mat = weight[i]
            min_ = min(mat.data)
            max_ = max(mat.data)
            space = np.linspace(min_, max_, num=2**bits)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
            kmeans.fit(mat.reshape(-1,1))
            cluster_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            cluster_weight[zero_index]=0.0

            if i == 0:
                new_weight = cluster_weight
            elif i == 1:
                new_weight = np.append([new_weight],[cluster_weight], axis=0)
            else:
                new_weight = np.append(new_weight,[cluster_weight], axis=0)
        new_weight = new_weight.reshape(shape)
        module.weight.data = torch.from_numpy(new_weight).to(dev)

        quantized_index = kmeans.labels_.reshape(shape)
        quantized_center = kmeans.cluster_centers_ 
        old_weight_list.append(old_weight)
        new_weight_list.append(new_weight)
        quantized_index_list.append(quantized_index)
        quantized_center_list.append(quantized_center)
    print(len(quantized_center_list))
    
    return old_weight_list, new_weight_list, quantized_index_list, quantized_center_list
'''