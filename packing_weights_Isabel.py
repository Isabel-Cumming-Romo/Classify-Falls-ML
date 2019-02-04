
import numpy as np
def pack_weights (w_1, w_2, w_3):
    #a=np.zeros(43, 1)
    #input_layer_size*layer_hidden_one_size + layer_hidden_one_size*layer_hidden_two_size +layer_hidden_two_size*output_layer_size,
    a=[0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0]
    for i in range(input_layer_size):
        print(i)
        for j in range(layer_hidden_one_size):
             a[i*layer_hidden_one_size+j]=w_1[i][j]
    
    for i in range(layer_hidden_one_size):
        for j in range(layer_hidden_two_size):
            a[input_layer_size*layer_hidden_one_size + i*layer_hidden_two_size + j]= w_2[i][j]
            
    for i in range(layer_hidden_two_size):
        for j in range(output_layer_size):
            a[input_layer_size*layer_hidden_one_size + layer_hidden_one_size*layer_hidden_two_size + i*output_layer_size + j]=w_3[i][j]
            
    return a

input_layer_size=5;
layer_hidden_one_size=5
layer_hidden_two_size=3
output_layer_size=1

w_1= [[1, 1, 1, 1,1],[1, 1, 1, 1,1],[1, 1, 1, 1,1],[1, 1, 1, 1,1],[1, 1, 1, 1,1]] 
w_2= [[2, 2, 2],[2, 2, 2],[2, 2, 2],[2, 2, 2],[2, 2, 2]]
w_3= [[3], [3], [3]]

a=[0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0]
a= pack_weights(w_1, w_2, w_3)
print(a)

