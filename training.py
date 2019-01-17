#Main program
import random # for w's initalizations
import numpy # for all matrix calculations
import math # for sigmoid


	#initialize lambda (if we're using lambda?)
	

	#initialize lambda
	lam=0
	
	#initialize max number of iterations
	max_iterations=5
	
	# Initialize weights to random numbers: w_1, w_2, w_3 ...
	w_1= numpy.matrix(numpy.random.random((2, 256))) #first make an nd-array, then convert that to a matrix
	w_2= numpy.matrix(numpy.random.random((256, 256)))
	w_3= numpy.matrix(numpy.random.random((256, 1)))
	
	#Import x (using pandas?)
	
	x_num_rows=3;
	

# for 1: max_iterarions
for x in range(max_iterations):

	#FEED-FORWARD PROPAGATION
		
		#layer1_activation =concatenate a column of all ones with X. 
		#all_ones = numpy.ones((x_num_rows,1)) #a column of 1's
		#layer1_activation= numpy.hstack((all_ones, X))# i.e. add a column of 1's to the front of the X matrix
		layer1_activation=X; #TODO-temporarily use layer1_activation without the bias (i.e. the column of 1's)
				#ie each row is a training example. The first column of each row is now a 1.
				#so you just add a column -like "the one" feature
				#layer1_activation is our first layer
			
		#z_2 = w_1 * layer1_activation	
		z_2 = numpy.matmul(w_1, layer1_activation) #intermediary variable
		
		#Compute layer2_activation = sigmoid(z_2)
		layer2_activation= sigmoid (z_2)
			
		#Compute a_3
			# Concatenate a bias column of all 1s with layer2_activation	
			# z_3 = w_3*layer2_activation
			#  layer3_activation = sigmoid(z_3)
	
		#Continue until the output_layer_activation, which should actually be h. 
		#The sigmoid result is your prediction (yay!)

		
		#BACKPROP
		
		#output_layer_gradient = 2*(output_layer_activation - y)/m
		
		#calculate gradients w.r.t. W3 and h2 (see defination of W3 and h2 in the figure of the handout) 
        #[W3_gradient, layer2_h_gradient] = compute_gradient_for_weights_and_one_layer_below(output_layer_gradient, W3, layer2_h);
		
		
		#

#ACTIVATION FUNCTION
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
		#Takes in a matrix
		#Computes 1/(1+ e^(-input))
		#Sanity check: this is done element-wise. It more or less changes the result to be between 0 and 1 
				#(actually closer to 0 or 1 because of the nature of the function)
				