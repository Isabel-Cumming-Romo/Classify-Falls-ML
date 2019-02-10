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
	
input_layer_size=2574
layer_hidden_one_size=500
layer_hidden_two_size=100
output_layer_size=1
	
	#initialize lambda
lam=1

	
data = pd.read_csv("capstoneData_norm.csv", low_memory=False);

data=numpy.random.permutation(data)

	#this is the number of features in the training matrix being read in (in the MATLAB code, is 256)
num_features=2574;
	
	#this is the number of samples (i.e. rows)
x_num_rows=600

input = data[:, 0:2574]; #the class labels are the last column of the csv file
output=numpy.matrix(data[:, 2574]).T;

print(numpy.shape(input))
print(output)


# Initialize weights to random numbers: w_1, w_2, w_3 ...# TO DO: make sure the initialization numbers are small (between 0 and 1)
w_1= numpy.matrix(numpy.random.random((input_layer_size, layer_hidden_one_size))) #
w_2= numpy.matrix(numpy.random.random((layer_hidden_one_size, layer_hidden_two_size)))
w_3= numpy.matrix(numpy.random.random((layer_hidden_two_size, output_layer_size)))

step=0

batch_size=20

lossHistory = []
loss=numpy.zeros(30)

for epoch in range(200):
	
	count=0
	
	for i in range(30):
	
		X=input[count:(count+batch_size), :]
		y=output[count:(count+batch_size), :]	
		
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
		
		loss[i]=cost
		
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
		
		# w_1 += numpy.dot(layer1_activation, W1_gradient)
		# w_2 += numpy.dot(layer2_activation, W2_gradient)
		# w_3 += numpy.dot(layer3_activation, W3_gradient)

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
		
	avg=numpy.sum(loss)/30
	loss=numpy.zeros(30)
	lossHistory.append(avg)	
	
layer1_activation=input
z_2 = numpy.dot(layer1_activation, w_1) 
layer2_activation= sigmoid (z_2)
		
z_3= numpy.dot(layer2_activation, w_2)
layer3_activation = sigmoid(z_3)
z_out= numpy.dot(layer3_activation, w_3)

h = sigmoid(z_out)
h[h >= 0.5]=1
h[h < 0.5]=0

cost=computeCost(input, output, h, x_num_rows)

print("Cost is " + str(cost))
print("Actual: " + str(output[100:110]))
print("Predicted: " + str(h[100:110]))
result=(h==output)
print("Accuracy is: " +str(numpy.sum(result)/x_num_rows))
print(sum(h))

plt.figure(figsize=(12, 9))
plt.subplot(311)
plt.plot(lossHistory)
#plt.subplot(312)
# plt.plot(H, '-*')
# plt.subplot(313)
# plt.plot(x, Y, 'ro')    # training data
# plt.plot(X[:, 1], Z, 'bo')   # learned
plt.show()

# print("w_1:")
# print(w_1)
# print("w_2:")
# print(w_2)
# print("w_3:")
# print(w_3)
