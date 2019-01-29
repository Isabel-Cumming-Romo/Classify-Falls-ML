#Main program
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
	
def pack_weights(w1, w2, w3):
    return np.concatenate((w1.reshape(-1), w2.reshape(-1), w3.reshape(-1)))
	
def unpack_weights_array (weights):
    w_1=numpy.empty([input_layer_size, layer_hidden_one_size])
    w_2=numpy.empty([layer_hidden_one_size, layer_hidden_two_size])
    w_3=numpy.empty([layer_hidden_two_size, output_layer_size])
    for i in range(input_layer_size):
        for j in range(layer_hidden_one_size):
            w_1[i, j]= a[i*j]
    
    for i in range(layer_hidden_one_size):
        for j in range(layer_hidden_two_size):
            w_2[i, j]= a[input_layer_size*layer_hidden_one_size + i*j]
            
    for i in range(layer_hidden_two_size):
        for j in range(output_layer_size):
            w_3[i, j]= a[input_layer_size*layer_hidden_one_size + layer_hidden_one_size*layer_hidden_two_size + i*j]
            
    return w_1, w_2, w_3
	
	# TO TEST UNPACK_WEIGHTS ARRAY: 
	#input_layer_size=5;
	#layer_hidden_one_size=5
	#layer_hidden_two_size=3
	#output_layer_size=1
	#a= [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
	#2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
	#3, 3, 3]
	#w_1, w_2, w_3 = unpack_weights_array(a)
	#print(w_1)
	#print(w_2)
	#print(w_3)
    

def computeCost(X, y, h, m):#Paramters: X, y, h (the hypothesis/prediction from the neural network), m (number of training examples)
	#Returns the cost 
	#Which is basically the weighted number of examples classified incorrectly 
	#i.e. we're summing up the amounts by which the predictor was off and then regularizing the number
	#Parameters: Theta matrices, y, h (predictions from feedforward propagation)
	#First get unregularized cost
		
	
	J=0 #initialize J
	#for i=1:m
	
	for i in range(x_num_rows):
	    
	    #J = J + y(i)*log(h(x(i)))+ (1-y(i))*log(1-h(x(i)))
	   
	    J = J + y[i]*(numpy.log(h[i])) + (1-y[i])*(numpy.log(1-h[i]))
	#end for
	#I think we can also do a simple vectorized implementation of y'*log(h) + (1-y)'*log(1-h)
			#but double check this
	#Divide J by -m
	J = J/(-x_num_rows)
		
	
	#Then regularize the cost by summing together each individual squared term of each w matrix 
	
	#Get rid of the first term of every w (this is the bias weight, we don't include it by convention, can try both ways)
	#don't do above^^ for now
	
	regTerm=numpy.sum(square(w_1)) + numpy.sum(square(w_2)) + numpy.sum(square(w_3))
		
	regTerm = (regTerm * lam)/(2*x_num_rows)
	J = J+regTerm
	
		
	#Return final cost: J= J + regTerm
	return J




def FFP (weights, X, y, x_num_rows):
#FEED-FORWARD PROPAGATION
	#layer1_activation =concatenate a column of all ones with X. 
	#all_ones = numpy.ones((x_num_rows,1)) #a column of 1's
	
	layer1_activation=X; #TODO-temporarily use layer1_activation without the bias (i.e. the column of 1's)
			#ie each row is a training example. The first column of each row is now a 1.
			#so you just add a column -like "the one" feature
			#layer1_activation is our first layer3
		
	#z_2 = w_1 * layer1_activation	
	z_2 = numpy.matmul(layer1_activation, w_1) #intermediary variable (note: order is important for the multiplication so that dimensions match up)

	#Compute layer2_activation = sigmoid(z_2)
	layer2_activation= sigmoid (z_2)
		
	#Compute a_3
		# Concatenate a bias column of all 1s with layer2_activation	
		#all_ones = numpy.ones((x_num_rows,1)) #column of 1's
		#layer2_activation= numpy.hstack((all_ones,layer2_activation3))# i.e. add a column of 1's to the front of the layer2_activation #TODO-temporarily use layer2_activation without the bias (i.e. the column of 1's)
		# z_3 = w_2*layer2_activation
	z_3= numpy.matmul (layer2_activation, w_2)
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

	#cost= computeCost(X, y, h, m) #m is the number of training examples 
	cost= computeCost(X, y, h, x_num_rows) #y is the vector containing the class values of the training data
	return cost
	
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

    #Now calculate gradient of layer 2
	#TO-DO: Remove first column of w_3
		
	W3_gradient, layer2_act_gradient = computeGradient(output_layer_gradient, w_3, layer3_activation)
	
	layer2_z_gradient = numpy.multiply(layer2_act_gradient, sigmoidGradient(z_3))
	#TO-DO: double check use of z vs. h here

	
	#Now for layer 1
	#TO-DO Remove first column of w_2
		
	#SITE OF BIG CHANGES 
	W2_gradient, layer1_act_gradient = computeGradient(layer2_act_gradient, w_2, layer2_activation)
	
	#Input layer
	layer1_z_gradient = numpy.multiply(layer1_act_gradient, sigmoidGradient(z_2))
	
	W1_gradient, throwAway = computeGradient(layer1_z_gradient, w_1, X)
	print(W1_gradient.shape)
	
	#TO-DO, return gradients in neccessary format
	return 0
	
#====START OF "MAIN"================
	
#data = pd.read_csv("CapstoneData_Revised.csv", low_memory=False);

	#this is the number of features in the training matrix being read in (in the MATLAB code, is 256)
#num_features=10299;
	
	#this is the number of samples (i.e. rows)
#x_num_rows=647;

#X = data.iloc[:, 10299]; #the class labels are the last column of the csv file
#Y=data.iloc[:, 0:10298];

#this is the number of features in the training matrix being read in (in the MATLAB code, is 256)
input_layer_size=3;
	
	#this is the number of samples (i.e. rows)
x_num_rows=10;

layer_hidden_one_size=5
layer_hidden_two_size=3
output_layer_size=1

#X = data.iloc[:, 10299]; #the class labels are the last column of the csv file
#Y=data.iloc[:, 0:10298];
	
	#initialize lambda
lam=1
	
	#initialize max number of iterations
max_iter=5
	
#for now, have X being a matrix that is 600Xnum_features big  filled with 5's
X= numpy.matrix(numpy.random.random((x_num_rows, input_layer_size)))
print("X initialized:")
print(X)
y=[1,1,0,1,0,0,0,1,1,0]
print("Y initialized:")
print(y)

# Initialize weights to random numbers: w_1, w_2, w_3 ...# TO DO: make sure the initialization numbers are small (between 0 and 1)
w_1= numpy.matrix(numpy.random.random((input_layer_size, layer_hidden_one_size))) #for now, since don't know what # of internal nodes will have (i.e. the latter dimension of this matrix), just make it 256
w_2= numpy.matrix(numpy.random.random((layer_hidden_one_size, layer_hidden_two_size)))
w_3= numpy.matrix(numpy.random.random((layer_hidden_two_size, output_layer_size)))
	

# for 1: max_iterarions
#for x in range(max_iterations):

    
	#Then we use a built-in optimizer 
			#We pass in X, y, the gradients, and potentially the cost function (FFP_BP)
			#We should receive updated weights back
			
			#model = opt.minimize(fun = CostFunc, x0 = initial_theta, args = (X, y), method = 'TNC', jac = Gradient)
options = {'maxiter': self.maxiter}
_res = optimize.minimize(self.function, thetas0, jac=self.function_prime, method=self.method,
args=(input_layer_size, self.hidden_layer_size, num_labels, X, y, 0), options=options)
self.t1, self.t2 = self.unpack_thetas(_res.x, input_layer_size, self.hidden_layer_size, num_labels)

				
