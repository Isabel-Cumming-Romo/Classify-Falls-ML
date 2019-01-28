import random # for w's initalizations
import numpy # for all matrix calculations
import math # for sigmoid
import scipy
import pandas as pd

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x)) 
	
def sigmoidGradient(z):
		#Parameters: z (a numerical input vector or matrix)
		#Returns: vector or matrix updated by
		 return numpy.multiply(sigmoid(z), (1-sigmoid(z)))

def computeGradient(upper_grad, w, X):
	# Return W_grad, h_grad
	#Params: upper_gradient (ie the gradient received from the layer above), W (the weight of one layer),
    #X (training data)
		
	W_grad = numpy.matmul(numpy.transpose(X), upper_grad)
	h_grad = numpy.matmul(upper_grad, numpy.transpose(w))
	return W_grad, h_grad
	
#BACKPROP
def backProp(X, y, x_num_rows):

	layer1_activation=X; #TODO-temporarily use layer1_activation without the bias (i.e. the column of 1's)
			#ie each row is a training example. The first column of each row is now a 1.
			#so you just add a column -like "the one" feature
			#layer1_activation is our first layer3
		
	#z_2 = w_1 * layer1_activation	
	z_2 = numpy.matmul(layer1_activation, w_1) #intermediary variable (note: order is important for the multiplication so that dimensions match up)

	#Compute layer2_activation = sigmoid(z_2)
	layer2_activation= sigmoid (z_2)
	print("Layer 1 activation shape is")
	print(layer2_activation.shape)
		
	#Compute a_3
		# Concatenate a bias column of all 1s with layer2_activation	
		#all_ones = numpy.ones((x_num_rows,1)) #column of 1's
		#layer2_activation= numpy.hstack((all_ones,layer2_activation3))# i.e. add a column of 1's to the front of the layer2_activation #TODO-temporarily use layer2_activation without the bias (i.e. the column of 1's)
		# z_3 = w_2*layer2_activation
	z_3= numpy.matmul (layer2_activation, w_2)
	print("Layer 2 activation shape is")
	print(z_3.shape)
		#  layer3_activation = sigmoid(z_3)
	layer3_activation = sigmoid(z_3)

	#Compute h (output layer activation...ie the hypothesis)
			#Concatenate bias column of all 1s with layer3_activation
			#all_ones = numpy.ones((x_num_rows,1)) #column of 1's
			#layer3_activation= numpy.hstack((all_ones,layer3_activation)) # i.e. add a column of 1's to the front of the layer3_activation #TODO-temporarily use layer3_activation without the bias (i.e. the column of 1's)
			# z_out = w_3*layer3_activation
	z_out= numpy.matmul (layer3_activation, w_3)
			# h = sigmoid(z_out)
	h = sigmoid(z_out)
	print("The prediction is\n")
	print(h)
	
		
	#Gradient of output layer
	output_layer_gradient = 2*numpy.subtract(h, y)/x_num_rows
	print("Output layer first. Gradient is size")
	print(output_layer_gradient.shape)

    #Now calculate gradient of layer 2
	#TO-DO: Remove first column of w_3
		
	W3_gradient, layer2_act_gradient = computeGradient(output_layer_gradient, w_3, layer3_activation)
	print("W3_grad:")
	print(W3_gradient.shape)
	print("layer2_act_grad:")
	print(layer2_act_gradient.shape)
	
	#In the ML prof's code, this was done element-wise, but that makes no sense to me
	#DOUBLE CHECK THIS
	#layer2_z_gradient = numpy.matmul(numpy.transpose(layer2_act_gradient), sigmoidGradient(layer2_activation))
	layer2_z_gradient = numpy.multiply(layer2_act_gradient, sigmoidGradient(layer3_activation))
	print("layer2_z_grad:")
	print(layer2_z_gradient.shape)
	
	#Now for layer 1
	#TO-DO Remove first column of w_2
		
	#SITE OF BIG CHANGES 
	W2_gradient, layer1_act_gradient = computeGradient(layer2_act_gradient, w_2, layer2_activation)
	
	#Input layer
	#DOUBLE CHECK THIS
	layer1_z_gradient = numpy.multiply(layer1_act_gradient, sigmoidGradient(layer2_activation))
	
	W1_gradient, throwAway = computeGradient(layer1_z_gradient, w_1, X)
	print(W1_gradient.shape)
	return 0

num_features=2;
	
	#this is the number of samples (i.e. rows)
x_num_rows=3;
	
#for now, have X being a matrix that is 600Xnum_features big  filled with 5's
X= numpy.matrix(numpy.random.random((x_num_rows, num_features)))
print("X initialized:")
print(X)
y=numpy.array([[1],[1],[0]])
print("Y initialized:")
print(y)

# Initialize weights to random numbers: w_1, w_2, w_3 ...# TO DO: make sure the initialization numbers are small (between 0 and 1)
w_1= numpy.matrix(numpy.random.random((num_features, 5))) #for now, since don't know what # of internal nodes will have (i.e. the latter dimension of this matrix), just make it 256
w_2= numpy.matrix(numpy.random.random((5, 4)))
w_3= numpy.matrix(numpy.random.random((4, 1)))
print(w_1.shape)
print(w_2.shape)
print(w_3.shape)

backProp(X,y,x_num_rows)


