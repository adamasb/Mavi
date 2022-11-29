
import torch
import numpy as np



n = 4
obs = torch.rand((32,n,n,3))

dim4 = [torch.ones((n,n,32))]
#dim4 = [torch.tensor([[1,2,3],[4,5,6],[7,8,9]]).reshape((3,3,1))] #has to be list of torch.size([4,4,1])



a_index = np.array([[1,1]]*32)


def get_neighborhood(obs,dim4,a_index):
    # print(obs.shape) #should be torch.size([32,n,n,3]), is torch.size([1,3,3,3])
    # print(dim4[0].shape) #should be torch.size([4,4,1]), is torch.size([3,3,1])
    # print(a_index[0]) #should be [x,y], is [1,1]


    print(obs.shape)
    print(len(dim4))
    print(dim4[0].shape)
    print(len(a_index))

    # torch.Size([32, 4, 4, 3])
    # 32
    # torch.Size([4, 4, 1])
    # 32
    
    # torch.Size([32, 4, 4, 3])
    # 32
    # torch.Size([4, 4, 1])
    # 32

    neighborhood = []
    v_neighborhood, w_neighborhood, a_neighborhood, g_neighborhood = [], [], [], []
    v_matrix, w_matrix, a_matrix,g_matrix = [], [], [], []

    #result = [ [] for _ in range(obs.shape[0]) ] 
    for ii in range(obs.shape[0]):
        v_matrix.append(torch.nn.functional.pad(dim4[ii].squeeze(),(1,1,1,1))) #dont wanna override tensors
        w_matrix.append(torch.nn.functional.pad(obs[ii][:,:,0],(1,1,1,1)))
        a_matrix.append(torch.nn.functional.pad(obs[ii][:,:,1],(1,1,1,1)))
        g_matrix.append(torch.nn.functional.pad(obs[ii][:,:,2],(1,1,1,1)))

        rowNumber = a_index[ii][0]+1 #numpy array so okay to override
        colNumber = a_index[ii][1]+1
        v_result, w_result, a_result, g_result = [], [], [], []

        results= []
        counter = 0
        for rowAdd in range(-1, 2):
            newRow = rowNumber + rowAdd
            if newRow >= 0 and newRow <= len(v_matrix[ii])-1:
                for colAdd in range(-1, 2):
                    newCol = colNumber + colAdd
                    if newCol >= 0 and newCol <= len(v_matrix[ii])-1:
                        if newCol == colNumber and newRow == rowNumber:
                            pass# this is the agent location itself
                            #continue
                        v_result.append(v_matrix[ii][newRow][newCol])                      
                        w_result.append(w_matrix[ii][newRow][newCol])
                        a_result.append(a_matrix[ii][newRow][newCol])
                        g_result.append(g_matrix[ii][newRow][newCol])
                        counter+=1

                        results.append(torch.tensor([v_matrix[ii][newRow][newCol],w_matrix[ii][newRow][newCol],a_matrix[ii][newRow][newCol],g_matrix[ii][newRow][newCol]]).flatten())

        print(counter)
        neighborhood.append(torch.tensor([w_result, a_result, g_result, v_result]).flatten())
        print(len(neighborhood[0]))
   
        # v_neighborhood.append(torch.tensor(v_result).reshape(3,3,1))
        # w_neighborhood.append(torch.tensor(w_result).reshape(3,3,1))
        # a_neighborhood.append(torch.tensor(a_result).reshape(3,3,1))
        # g_neighborhood.append(torch.tensor(g_result).reshape(3,3,1))


        
        # print(torch.tensor(v_neighborhood).shape)
        # print(torch.tensor(w_neighborhood).shape)
        # print(torch.tensor(a_neighborhood).shape)
        # print(torch.tensor(g_neighborhood).shape)
    # test = torch.stack(neighborhood)
    # v,w,a,g = torch.moveaxis(torch.stack(v_neighborhood).squeeze(),0,-1), torch.moveaxis(torch.stack(w_neighborhood).squeeze(),0,-1), torch.moveaxis(torch.stack(a_neighborhood).squeeze(),0,-1), torch.moveaxis(torch.stack(g_neighborhood).squeeze(),0,-1)

    # #v,w,a,g = torch.tensor(v_neighborhood).reshape((3,3,obs.shape[0])), torch.tensor(w_neighborhood).reshape((3,3,obs.shape[0])), torch.tensor(a_neighborhood).reshape((3,3,obs.shape[0])), torch.tensor(g_neighborhood).reshape((3,3,obs.shape[0]))

    # neighborhood = torch.cat((v,w,a,g),dim=2)
    
    # print(neighborhood[:,:,0])
    return torch.stack(neighborhood)
#vp is shape ([32,64])




#
dim4 = [torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]]), torch.torch.tensor([[[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         [0.]]])]

obs = torch.ones((32,n,n,3))

a_index = np.array([[0,0]]*32)


neighborhood = get_neighborhood(obs,dim4,a_index)
print(neighborhood[:,:,0])



"""

import torch
import numpy as np




obs = torch.zeros((3,3,3))
dim4 = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])

a_index = np.array([1,1])


def get_neighborhood(obs,dim4,a_index):
    neighbourhood = []
    # v_matrix = []
    # w_matrix = []
    # a_matrix = []
    # g_matrix = []

    result = [ [] for _ in range(obs.shape[0]) ] 
    # for ii in range(obs.shape[0]): maybe just do the whole thing for 1 batch only
    v_matrix = (torch.nn.functional.pad(dim4,(1,1,1,1))) #dont wanna override tensors
    w_matrix = (torch.nn.functional.pad(obs[:,:,0],(1,1,1,1)))
    a_matrix = (torch.nn.functional.pad(obs[:,:,1],(1,1,1,1)))
    g_matrix = (torch.nn.functional.pad(obs[:,:,2],(1,1,1,1)))


    rowNumber = a_index[0] #numpy array so okay to override
    colNumber = a_index[1]
    result = []
    for rowAdd in range(-1, 2):
        newRow = rowNumber + rowAdd
        if newRow >= 0 and newRow <= len(v_matrix)-1:
            for colAdd in range(-1, 2):
                newCol = colNumber + colAdd
                if newCol >= 0 and newCol <= len(v_matrix)-1:
                    # if newCol == colNumber and newRow == rowNumber:
                    #     continue
                    result.append(v_matrix[newCol][newRow])
                    # result.append(w_matrix[newCol][newRow])
                    # result.append(a_matrix[newCol][newRow])
                    # result.append(g_matrix[newCol][newRow])
                    a=1
        else:
            pass
        

        neighbourhood.append(result)
        return torch.tensor(neighbourhood)

neighbourhood = get_neighborhood(obs,dim4,a_index)
"""