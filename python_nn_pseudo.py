#Training the network

	#




#BACK PROPAGATION FUNCTION
		#Backpropagation algorithm to compute derivatives with respect to parameters
		
		#We're working with deltas, which are accumulators
		
		#For i=1:m
			#Set a_1 = x(i) Note: a_1 is the "activation" of the first layer (nothing is really activated, just use the notation for consistency)
			#Perform forward propagation (same as function below but for one example at a time. 
				#it really needs to be done separately (the function is for predicting)
					#Keep track of all a_Ls (the activiation for each layer L)
					
			# Compute delta_N = a_N - y(i) (where N is the last layer)
				#delta_N is simply the difference between the prediction and the real values
				
			#Then compute delta_N-1, delta_N-2, delta_N-3... until delta_2 (we don't do the computation for the first layer)
				#ie if L= N-1, N-2,...2
				
				
				#delta_L = (Theta_L)' * delta_L+1 .*sigmoidGradient(z_L)
						#So, multiply the transpose of the theta matrix for that layer by the delta matrix of the layer above
							#Then multiply element-wise by the sigmoid gradient (see function)
						#What's happening?
							#Well, we're kind of finding out how much error was introduced in each layer of our network
							#Formally, each delta is the partial derivative with respect to z_l,j of cost(i) 
									#remember we used z in forward propagation (we stuck it in the sigmoid function), i is each training example, l is each layer, j is each node
							#A measure of how much we want to change the NN's weights)
								
				#We end up with the gradients, which we need for search optimization
			
				#Theta1_grad = Theta1_grad* delta_2*a_1
				#Theta2_grad = Theta2_grad * delta_3*a_2
				#Repeat for all layers (i.e. all ThetaL_grads)
				
		# end loop
		
		#Theta1_grad = Theta1_grad/m + (lambda/m)*Theta1
		#Theta2_grad = Theta2_grad/m + (lamda/m)*Theta2
		#Do for all thetas

		#Return the gradients!

#SIGMOIDGRADIENT FUNCTION
	`#Takes one parameter z -> a matrix or a vector 
	#g= sigmoid(z).*(1-sigmoid(z))
	#i.e. it's just the element-wise multiplication of the sigmoid minus 1-sigmoid
	#(take derivative to derive this formula if unconvinced)




#COMPUTE THE COSTFUNCTION FUNCTION
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
			

#FEED-FORWARD PROPAGATION/PREDICTION FUNCTION 
	#Returns a class prediction for every example

	#parameters: X, Theta1, Theta2, Theta3... ThetaN (where N is number_of_layers -1)
	
	#a_1 =concatenate a column of all ones with X. 
			#ie each row is a training example. The first column of each row is now a 1.
			#so you just add a column -like "the one" feature
			#a_1 is our first layer
			
	#Compute a_2 (activation of layer 2)
			# This is simply activationFunction(Theta1*a_1) => usually use Sigmoid
			
	#Compute a_3
			# Concatenate a bias column of all 1s with a_2	
			# activationFunction(Theta2*a_2)
	
	#Continue until the output layer (a_(N+1)). 
		#The sigmoid result is your prediction (yay!)
		
	#Call COSTFUNCTION FUNCTION 
	
	#Return cost
		
#ACTIVATION FUNCTION

		#Takes in a matrix
		#Computes 1/(1+ e^(-input))
		#Sanity check: this is done element-wise. It more or less changes the result to be between 0 and 1 
				#(actually closer to 0 or 1 because of the nature of the function)
				