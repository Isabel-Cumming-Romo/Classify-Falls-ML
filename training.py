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
	
<<<<<<< HEAD
	#Initialize y=zeros(m, 1)
	
=======
	x_num_rows=3;
	

>>>>>>> a5583e0afe00b3a02c7b976882eb677664b91b6d
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
			# z_3 = w_2*layer2_activation
			#  layer3_activation = sigmoid(z_3)
	
		#Compute h (output layer activation...ie the hypothesis)
				#Concatenate bias column of all 1s with layer3_activation
				# z_out = w_3*layer3_activation
				# h = sigmoid(z_out)

		#cost= computeCost(X, y, h, m)
			#m is the number of training examples 
		
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
    
	#Then we use a built-in optimizer 
			#We pass in X, y, the gradients, and potentially the cost function
			#We should receive updated weights back
			
			#model = opt.minimize(fun = CostFunc, x0 = initial_theta, args = (X, y), method = 'TNC', jac = Gradient)
		
#COMPUTE GRADIENT FUNCTION

		# Return W_grad, h_grad
		#Params: upper_gradient (ie the gradient received from the layer above), W, X (in this order)
		
		# W_grad = X' * upper_gradient; where ' means transpose 
		
		# h_grad = upper_gradient * W';
		
		
#activationGrad FUNCTION

		#Parameters: z (a numerical input vector or matrix)
		#Returns: vector or matrix updated by
			#sigmoid(z).*(1-sigmoid(z))
		

#ACTIVATION FUNCTION
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
		#Takes in a matrix
		#Computes 1/(1+ e^(-input))
		#Sanity check: this is done element-wise. It more or less changes the result to be between 0 and 1 
				#(actually closer to 0 or 1 because of the nature of the function)
				
#computeCost(X, y, h, m)
	#Paramters: X, y, h (the hypothesis/prediction from the neural network), m (number of training examples)
		#Returns the cost 
		#Which is basically the weighted number of examples classified incorrectly 
		#i.e. we're summing up the amounts by which the predictor was off and then regularizing the number
		#Parameters: Theta matrices, y, h (predictions from feedforward propagation)
		#First get unregularized cost
			
			#for i=1:m
			#J = J + y(i)*log(h(x(i)))+ (1-y(i))*log(1-h(x(i)))
			#end for
			#I think we can also do a simple vectorized implementation of y'*log(h) + (1-y)'*log(1-h)
					#but double check this
			#Divide J by -m
			
		
		#Then regularize the cost by summing together each individual squared term of each theta matrix 
		
			#Get rid of the first term of every Theta (this is the bias weight, we don't include it by convention, can try both ways)
			#regTerm = sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)) + ...+ sum(sum(ThetaN.^2)
			#regTerm = (regTerm * lambda)/2/m
			
		#Return final cost: J= J + regTerm
				