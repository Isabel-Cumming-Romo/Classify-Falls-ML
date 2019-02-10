import random # for w's initalizations
import numpy # for all matrix calculations
import math # for sigmoid
import scipy
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x)) 
	
def sigmoidGradient(z):
		#Parameters: z (a numerical input vector or matrix)
		#Returns: vector or matrix updated by
		 return numpy.multiply(sigmoid(z), (1-sigmoid(z)))

def square(x):
    return numpy.power(x, 2)
	

def computeCost(X, y, h, m):#Paramters: X, y, h (the hypothesis/prediction from the neural network), m (number of training examples)

	# J=0 #initialize J
	# for i in range(x_num_rows):
		# #J = J + y(i)*log(h(x(i)))+ (1-y(i))*log(1-h(x(i)))
	   
		# J = J + y[i]*(numpy.log(h[i])) + (1-y[i])*(numpy.log(1-(h[i])))
		# s=numpy.shape(h)

		# J=J/(-s[0])
		
	
	# #Then regularize the cost by summing together each individual squared term of each w matrix 
	
	# #Get rid of the first term of every w (this is the bias weight, we don't include it by convention, can try both ways)
	# #don't do above^^ for now
	
	# regTerm=numpy.sum(square(w_1)) + numpy.sum(square(w_2)) + numpy.sum(square(w_3))
		
	# regTerm = (regTerm * lam)/(2*x_num_rows)
	# J = J+regTerm
	# print(J)
	J=numpy.sum(numpy.square(y-h))
	
	s=numpy.shape(h)

	J=J/s[0]

	return J
	
def computeGradient(upper_grad, w, X):
	# Return W_grad, h_grad
	#Params: upper_gradient (ie the gradient received from the layer above), W (the weight of one layer),
    #X (training data)
		
	W_grad = numpy.matmul(numpy.transpose(X), upper_grad)
	h_grad = numpy.matmul(upper_grad, numpy.transpose(w))
	return W_grad, h_grad
	
input_layer_size=3
layer_hidden_one_size=10
layer_hidden_two_size=20
output_layer_size=1
x_num_rows=4
	
	#initialize lambda
lam=1

	
X = numpy.matrix([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = numpy.matrix([[0,0,1,1]]).T	
	
# data = pd.read_csv("CapstoneData_Revised.csv", low_memory=False);

# data=numpy.random.permutation(data)

	# #this is the number of features in the training matrix being read in (in the MATLAB code, is 256)
# num_features=10299;
	
	# #this is the number of samples (i.e. rows)
# x_num_rows=50;

# output = numpy.matrix(data[0:650, 10299]).T #the class labels are the last column of the csv file
# input=data[0:650, 0:10299]

# test=data[650:673, :]
# Xtest=test[:, 0:10299]
# ytest=test[:, 10299]

# max=numpy.amax(input,0)
# min=numpy.amin(input,0)
# print(numpy.shape(max))
# for i in range(10298): 
	# avg=numpy.sum(input[:,i])/650
	# for l in range(649):
		# input[l,i]=(input[l,i]-avg)/(max[i]-min[i])



# Initialize weights to random numbers: w_1, w_2, w_3 ...# TO DO: make sure the initialization numbers are small (between 0 and 1)
w_1= numpy.matrix(numpy.random.random((input_layer_size, layer_hidden_one_size)))
w_2= numpy.matrix(numpy.random.random((layer_hidden_one_size, layer_hidden_two_size)))
w_3= numpy.matrix(numpy.random.random((layer_hidden_two_size, output_layer_size)))


# print("w_1:")
# print(numpy.shape(w_1))
# print("w_2:")
# print(numpy.shape(w_2))
# print("w_3:")
# print(numpy.shape(w_3))

step=0

batch_size=50
lossHistory = []
testHistory=[]

print("Training neural network")
for epoch in range(1):
	
	count=0
	
	for i in range(100):
		
		#X=input[count:(count+batch_size), :]
	
	#	y=output[count:(count+batch_size), :]	
		
		m_W1=0
		v_W1=0
		m_W2=0
		v_W2=0
		m_W3=0
		v_W3=0
		
		#Forward propagation
		layer1_activation=X; 
		#print("X: " + str(numpy.shape(X)))
		z_2 = numpy.dot(layer1_activation, w_1) 
		#print("z_2: " + str(numpy.shape(z_2)))
		layer2_activation= sigmoid (z_2)
			
		z_3= numpy.dot(layer2_activation, w_2)
		#print("z_3: " + str(numpy.shape(z_3)))
		layer3_activation = sigmoid(z_3)
		
		z_4= numpy.dot(layer3_activation, w_3)
		#print("z_4: " + str(numpy.shape(z_4)))
		h = sigmoid(z_4)
		#print("h: " + str(numpy.shape(h)))
		cost=computeCost(X, y, h, x_num_rows)
		
		lossHistory.append(cost)
		
		#print("Iteration " + str(i))
		#print("Cost is " + str(cost))
		
		
		#Back Propagation
		output_layer_gradient = 2*numpy.subtract(h, y)/x_num_rows
		
		W3_gradient, layer2_act_gradient = computeGradient(output_layer_gradient, w_3, layer3_activation)
		
		layer2_z_gradient = numpy.multiply(layer2_act_gradient, sigmoidGradient(z_3))

		W2_gradient, layer1_act_gradient = computeGradient(layer2_act_gradient, w_2, layer2_activation)
		
		#Input layer
		layer1_z_gradient = numpy.multiply(layer1_act_gradient, sigmoidGradient(z_2))
		
		W1_gradient, throwAway = computeGradient(layer1_z_gradient, w_1, X)

		step = step+1 #step + 1
		m_W1 = (0.9 * m_W1 + 0.1 * W1_gradient)
		v_W1 = (0.999 * v_W1 + 0.001 * numpy.square(W1_gradient))
		w_1 = w_1 - 0.01 * numpy.divide((m_W1/(1-(0.9**step))), numpy.sqrt(v_W1/(1-(0.999**step)) + 1e-8))
		
		m_W2 = (0.9 * m_W2 + 0.1 * W2_gradient)
		v_W2 = (0.999 * v_W2 + 0.001 * numpy.square(W2_gradient))
		w_2 = w_2 - 0.01 * numpy.divide((m_W2/(1-(0.9**step))), numpy.sqrt(v_W2/(1-(0.999**step)) + 1e-8))
		
		m_W3 = (0.9 * m_W3 + 0.1 * W3_gradient)
		v_W3 = (0.999 * v_W3 + 0.001 * numpy.square(W3_gradient))
		w_3 = w_3 - 0.01 * numpy.divide((m_W3/(1-(0.9**step))), numpy.sqrt(v_W3/(1-(0.999**step)) + 1e-8))
		
		count=count+batch_size
		
layer1_activation=X; 
#layer1_activation=input; 
z_2 = numpy.dot(layer1_activation, w_1) 
layer2_activation= sigmoid (z_2)
		
z_3= numpy.dot(layer2_activation, w_2)
layer3_activation = sigmoid(z_3)
z_out= numpy.dot(layer3_activation, w_3)

h = sigmoid(z_out)
h[h >= 0.5]=1
h[h < 0.5]=0

#cost=computeCost(input, output, h, x_num_rows)
cost=computeCost(X, y, h, x_num_rows)

print("Cost is " + str(cost))

#result=(h==output)
# print(str(numpy.sum(result)/650))
# print(str(sum(h)))

print("Actual: " + str(y))
print("Predicted: " + str(h))



plt.figure(figsize=(12, 9))
plt.subplot(311)
plt.plot(lossHistory)
#plt.plot(testHistory, 'b')
#plt.subplot(312)
# plt.plot(H, '-*')
# plt.subplot(313)
# plt.plot(x, Y, 'ro')    # training data
# plt.plot(X[:, 1], Z, 'bo')   # learned
plt.show()

