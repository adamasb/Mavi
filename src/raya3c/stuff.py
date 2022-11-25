
import torch
import numpy as np




def neb(s, dir):
    neighbourhood = []
    matrix = []
    neib= []
    result = [ [] for _ in obs.shape[0] ] 
    for ii in range(obs.shape[0]):
        matrix.append(torch.nn.functional.pad(vp[ii],(1,1,1,1))) #dont wanna override tensors
        rowNumber = a_index[ii][0] #numpy array so okay to override
        colNumber = a_index[ii][1]
        result = []
        for rowAdd in range(-1, 2):
            newRow = rowNumber + rowAdd
            if newRow >= 0 and newRow <= len(matrix[ii])-1:
                for colAdd in range(-1, 2):
                    newCol = colNumber + colAdd
                    if newCol >= 0 and newCol <= len(matrix[ii])-1:
                        if newCol == colNumber and newRow == rowNumber:
                            continue
                        result.append(matrix[ii][newCol][newRow])