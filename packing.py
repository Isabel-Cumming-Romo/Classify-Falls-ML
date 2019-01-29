import numpy 

def pack_weights(w1, w2, w3):
	
	size=input_layer_size*layer_hidden_one_size + layer_hidden_one_size*layer_hidden_two_size + layer_hidden_two_size*output_layer_size
	weights=numpy.zeros(size)
	print("size is ")
	print(size)

	
	i=0
		
	for k in range(input_layer_size):
		for j in range(layer_hidden_one_size):
			weights[i]=w1[k, j]
			i=i+1
	#print(weights)			
	for k in range(layer_hidden_one_size):
		for j in range(layer_hidden_two_size):
			weights[i]=w2[k, j]	
			i=i+1
	#print(weights)	
	for k in range(layer_hidden_two_size):
		weights[i]=w3[k, 0]	
		i=i+1
			
	print(weights)
	return weights
		
	
input_layer_size=5
layer_hidden_one_size=9
layer_hidden_two_size=3
output_layer_size=1

w_1= numpy.matrix(numpy.random.random((input_layer_size, layer_hidden_one_size))) #for now, since don't know what # of internal nodes will have (i.e. the latter dimension of this matrix), just make it 256
w_2= numpy.matrix(numpy.random.random((layer_hidden_one_size, layer_hidden_two_size)))
w_3= numpy.matrix(numpy.random.random((layer_hidden_two_size, output_layer_size)))

print("w1")
print(w_1)
print("w2")
print(w_2)
print("w3")
print(w_3)

weights=pack_weights(w_1, w_2, w_3)


