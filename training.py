#Main program
import random # for w's initalizations
import numpy # for all matrix calculations
import math # for sigmoid
import scipy
import pandas as pd

def FFP_BP (x):
#FEED-FORWARD PROPAGATION
	#layer1_activation =concatenate a column of all ones with X. 
	#all_ones = numpy.ones((x_num_rows,1)) #a column of 1's
	print("in FFP")
	
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

	#cost= computeCost(X, y, h, m) #m is the number of training examples 
	cost= computeCost(X, y, h, x_num_rows) #y is the vector containing the class values of the training data
		
#BACKPROP
		
		#output_layer_gradient = 2*(output_layer_activation - y)/m
		
		#calculate gradients w.r.t. W3 and h2 (see defination of W3 and h2 in the figure of the handout) 
        
		#Remove first column of w_3
		
		#[W3_gradient, layer2_act_gradient] = compute_gradient(output_layer_gradient, w_3, layer2_activation);
		
		
		#layer2_z_gradient = layer2_act_gradient.*activationGrad(layer2_activation);
		
		#Remove first column of w_2
		
		#[W2_gradient, layer1_act_gradient] = compute_gradient(layer2_act_gradient, w_2, layer1_activation);

        
        #layer1_z_gradient = layer1_act_gradient.* activationGrad(layer1_activation);
		
		#Remove first column of w_1
 
        #[W1_gradient, ~] = compute_gradient(layer1_z_gradient, w_1, X);

	return cost
	
#COMPUTE GRADIENT FUNCTION

		# Return W_grad, h_grad
		#Params: upper_gradient (ie the gradient received from the layer above), W, X (in this order)
		
		# W_grad = X' * upper_gradient; where ' means transpose 
		
		# h_grad = upper_gradient * W';
		
		
#activationGrad FUNCTION

		#Parameters: z (a numerical input vector or matrix)
		#Returns: vector or matrix updated by
			#sigmoid(z).*(1-sigmoid(z))
				


# Compute gradient and cost function 

	#return ans; 
	
	
#STILL WORKING ON THIS. BELOW CODE IS SHELL NEEDED FOR .PY INTERPRETER==============================================================================

import numpy
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x)) 
    
def square(x):
    return numpy.power(x, 2)
    
x_num_rows=3
num_features=256;

X= numpy.full((x_num_rows, num_features), -2)
print(numpy.size(X,0))
print(numpy.size(X,1))
y = numpy.ones((x_num_rows,1)) 
print(numpy.size(y,0))
print(numpy.size(y,1))

w= numpy.matrix(numpy.random.random((num_features, 1))) 
print(numpy.size(w,0))
print(numpy.size(w,1))
layer_activation=X;
z = numpy.matmul(layer_activation, w)
print(numpy.size(z,0))
print(numpy.size(z,1))
h= sigmoid (z)
#print(h[0,:][:,0])
print(numpy.size(h,0))
print(numpy.size(h,1))
print(h)
print("=========================================")
a= numpy.matrix([[1, 2], [4, 3]])
print(a)
a= square(a)
print(a.sum())

def computeCost(X, y, h, m):#Paramters: X, y, h (the hypothesis/prediction from the neural network), m (number of training examples)
	#Returns the cost 
	#Which is basically the weighted number of examples classified incorrectly 
	#i.e. we're summing up the amounts by which the predictor was off and then regularizing the number
	#Parameters: Theta matrices, y, h (predictions from feedforward propagation)
	#First get unregularized cost
		
	
	J=0 #initialize J
	#for i=1:m
	for i in range(x_num_rows):
	    #print(h[i,:][:,0]) #this is how print each entry in the vector h
	    #print(y[i]) #this is how print each entry in the array y
	    #J = J + y(i)*log(h(x(i)))+ (1-y(i))*log(1-h(x(i)))
	    print("hi")
	    J = J + y[i]*log(h[i,:][:,0]) + (1-y[i])*log(1-h[i,:][:,0])
	#end for
	#I think we can also do a simple vectorized implementation of y'*log(h) + (1-y)'*log(1-h)
			#but double check this
	#Divide J by -m
	J=J/-x_num_rows
		
	
	#Then regularize the cost by summing together each individual squared term of each w matrix 
	
	#Get rid of the first term of every w (this is the bias weight, we don't include it by convention, can try both ways)
	#don't do above^^ for now
	#regTerm = sum(sum(w_1.^2)) + sum(sum(w_2.^2)) + ...+ sum(sum(w_N.^2)
	#i.e. sum up ALL of the weights' squares
		
	#regTerm = (regTerm * lambda)/2/m
		
	#Return final cost: J= J + regTerm
	return J

computeCost(X, y, h, x_num_rows)
#print(y)
#print(X)
#print(computeCost(X, y, h, x_num_rows))
#print("==============================")
#print(h)


#data = pd.read_csv("CapstoneData_Revised.csv", low_memory=False);

	#this is the number of features in the training matrix being read in (in the MATLAB code, is 256)
num_features=10299;
	
	#this is the number of samples (i.e. rows)
x_num_rows=647;

#X = data.iloc[:, 10299]; #the class labels are the last column of the csv file
#Y=data.iloc[:, 0:10298];
	
	#initialize lambda
lam=0
	
	#initialize max number of iterations
max_iterations=5
	
	# Initialize weights to random numbers: w_1, w_2, w_3 ...# TO DO: make sure the initialization numbers are small (between 0 and 1)
w_1= numpy.matrix(numpy.random.random((num_features, num_features))) #for now, since don't know what # of internal nodes will have (i.e. the latter dimension of this matrix), just make it 256
w_2= numpy.matrix(numpy.random.random((num_features, num_features)))
w_3= numpy.matrix(numpy.random.random((num_features, 1)))
	
	
	#for now, have X being a matrix that is 600Xnum_features big  filled with 5's
X= numpy.full((x_num_rows, num_features), 2)
print("X initialized")
	#Initialize y=zeros(m, 1)



# for 1: max_iterarions
#for x in range(max_iterations):

    
	#Then we use a built-in optimizer 
			#We pass in X, y, the gradients, and potentially the cost function (FFP_BP)
			#We should receive updated weights back
			
			#model = opt.minimize(fun = CostFunc, x0 = initial_theta, args = (X, y), method = 'TNC', jac = Gradient)
FFP_BP(x=X);
				
