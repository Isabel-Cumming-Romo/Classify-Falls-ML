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

def square(x):
    return numpy.power(x, 2)
	

def computeCost(X, y, h, m):#Paramters: X, y, h (the hypothesis/prediction from the neural network), m (number of training examples)

	J=numpy.sum(numpy.square(y-h))
	
	s=numpy.shape(h)

	J=J/s[0]
		
	#Return final cost: J= J + regTerm
	return J

	
def computeGradient(upper_grad, w, X):
	# Return W_grad, h_grad
	#Params: upper_gradient (ie the gradient received from the layer above), W (the weight of one layer),
    #X (training data)
		
	W_grad = numpy.matmul(numpy.transpose(X), upper_grad)
	h_grad = numpy.matmul(upper_grad, numpy.transpose(w))
	return W_grad, h_grad
	
	
input_layer_size=2;
	
	#this is the number of samples (i.e. rows)
x_num_rows=3;

layer_hidden_one_size=4
#layer_hidden_two_size=4
output_layer_size=1
	
	#initialize lambda
lam=1
	
	#initialize max number of iterations
max_iter=5
	

X = numpy.array([[0,0],
                [0,1],
                [1,1]])
    
# output dataset            
y = numpy.array([[0,0,1]]).T	


# Initialize weights to random numbers: w_1, w_2, w_3 ...# TO DO: make sure the initialization numbers are small (between 0 and 1)
w_1= numpy.matrix(numpy.random.random((input_layer_size, layer_hidden_one_size))) #
w_2= numpy.matrix(numpy.random.random((layer_hidden_one_size, output_layer_size)))
#w_3= numpy.matrix(numpy.random.random((layer_hidden_two_size, output_layer_size)))

for i in range(1000):
	
	m_W1=0
	v_W1=0
	m_W2=0
	v_W2=0
	
	#Forward propagation
	layer1_activation=X; 
	#print("X: " + str(numpy.shape(X)))
	z_2 = numpy.dot(layer1_activation, w_1) 
	#print("z_2: " + str(numpy.shape(z_2)))
	layer2_activation= sigmoid (z_2)
		
	z_3= numpy.dot(layer2_activation, w_2)
	#print("z_3: " + str(numpy.shape(z_3)))
	h = sigmoid(z_3)

	#print("h: " + str(numpy.shape(h)))
	cost=computeCost(X, y, h, x_num_rows)
	
	#print("Iteration " + str(i))
	print("Cost is " + str(cost))
	
	
	#Back Propagation
	output_layer_gradient = 2*numpy.subtract(h, y)/x_num_rows
	
	W2_gradient, layer1_act_gradient = computeGradient(output_layer_gradient, w_2, layer2_activation)
	
	#Input layer
	layer1_z_gradient = numpy.multiply(layer1_act_gradient, sigmoidGradient(z_2))
	
	W1_gradient, throwAway = computeGradient(layer1_z_gradient, w_1, X)
	print("W1_gradient " + str(numpy.shape(W1_gradient))) # 2 x 5
	print("X " + str(numpy.shape(layer1_activation))) # 3 x 2
	print("w_1: " + str(numpy.shape(w_1))) # 2 x 5
	
	# w_1 += numpy.dot(layer1_activation, W1_gradient)
	# w_2 += numpy.dot(layer2_activation, W2_gradient)
	# w_3 += numpy.dot(layer3_activation, W3_gradient)

	step = 1 #step + 1
	m_W1 = (0.9 * m_W1 + 0.1 * W1_gradient)
	v_W1 = (0.999 * v_W1 + 0.001 * numpy.square(W1_gradient))
	w_1 = w_1 - 0.01 * numpy.divide((m_W1/(1-(0.9**step))), numpy.sqrt(v_W1/(1-(0.999**step)) + 1e-8))
	
	m_W2 = (0.9 * m_W2 + 0.1 * W2_gradient)
	v_W2 = (0.999 * v_W2 + 0.001 * numpy.square(W2_gradient))
	w_2 = w_2 - 0.01 * numpy.divide((m_W2/(1-(0.9**step))), numpy.sqrt(v_W2/(1-(0.999**step)) + 1e-8))
	
	# m_W3 = (0.9 * m_W3 + 0.1 * W3_gradient)
	# v_W3 = (0.999 * v_W3 + 0.001 * numpy.square(W3_gradient))
	# w_3 = w_3 - 0.01 * numpy.divide((m_W3/(1-(0.9**step))), numpy.sqrt(v_W3/(1-(0.999**step)) + 1e-8))
	
	
	
layer1_activation=X; 
z_2 = numpy.dot(layer1_activation, w_1) 
layer2_activation= sigmoid (z_2)
		
z_3= numpy.dot(layer2_activation, w_2)
h = sigmoid(z_3)

cost=computeCost(X, y, h, x_num_rows)
print("Cost is " + str(cost))
print(h)

# print("w_1:")
# print(w_1)
# print("w_2:")
# print(w_2)
# print("w_3:")
# print(w_3)
