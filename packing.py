import numpy 

def pack_weights(w1, w2, w3):

	

	return weights
		
def unpack_weights(weights, input_layer_size, hidden_layer_one_size, hidden_layer_two_size, num_labels):
	w1_start = 0
	w1_end = hidden_layer_one_size * (input_layer_size)
	w2_end = hidden_layer_two_size * hidden_layer_one_size
	w1 = weights[w1_start:w1_end].reshape((hidden_layer_one_size, input_layer_size))
	w2 = weights[w1_end:w2_end].reshape((hidden_layer_two_size, hidden_layer_one_size))
	w3 = weight[w2_end:].reshape((num_labels, hidden_layer_two_size))
	return w1, w2, w3
	
input_layer_size=5
layer_hidden_one_size=5
layer_hidden_two_size=3
output_layer_size=1

w_1= numpy.matrix(numpy.random.random((input_layer_size, layer_hidden_one_size))) #for now, since don't know what # of internal nodes will have (i.e. the latter dimension of this matrix), just make it 256
w_2= numpy.matrix(numpy.random.random((layer_hidden_one_size, layer_hidden_two_size)))
w_3= numpy.matrix(numpy.random.random((layer_hidden_two_size, output_layer_size)))

weights=numpy.column_stack(w_1, w_2)


