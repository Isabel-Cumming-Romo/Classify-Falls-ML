#Main program

	#initialize lambda (if we're using lambda?)
	
	#initialize max number of iterations
	
	# Initialize weights to random numbers: w_1, w_2, w_3 ...
	
	#We have X 
	
	#Initialize y=zeros(m, 1)

# for 1: max_iterarions

	#FEED-FORWARD PROPAGATION
		
		#layer1_activation =concatenate a column of all ones with X. 
				#ie each row is a training example. The first column of each row is now a 1.
				#so you just add a column -like "the one" feature
				#a_1 is our first layer
			
		#z_2 = w_2 * layer1_activation	
		#Compute layer2_activation = sigmoid(z_2)
			
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
		
		
		#...

#ACTIVATION FUNCTION

		#Takes in a matrix
		#Computes 1/(1+ e^(-input))
		#Sanity check: this is done element-wise. It more or less changes the result to be between 0 and 1 
				#(actually closer to 0 or 1 because of the nature of the function)
				