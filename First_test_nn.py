import tensorflow as tf #importing the tensorflow library
import pandas as pd #import csv file library

#WHERE ARE WE? => TYPE ERROR
#ValueError: Tensor conversion requested dtype float64 for 
# Tensor with dtype float32: 'Tensor("Variable/read:0", shape=(11792, 1), dtype=float32)

bias = 1.0

# Read the file
data = pd.read_csv("CapstoneData.csv", low_memory=False)

# Output the number of rows
print("Total rows: {0}".format(len(data)))

training_output = data.iloc[:, 11791] #the class labels are the last column of the csv file
training_output=tf.cast(training_output, tf.float32) #change the class labels (all 0 or 1) to be floating pt numbers (note: chose 32 because random_normal below only uses 32)
data.iloc[:, 0:11791]=bias #introduce bias unit- Make the last column be the bias (convention?)
training_input=tf.cast(data, tf.float32) #the training input is all the data + the last column (which is now the bias)
 
w = tf.Variable(tf.random_normal([11792, 1]), dtype=tf.float32) #instantiate the weights vector randomly

# step(x) = { 1 if x > 0; -1 otherwise }

def step(x):

    is_greater = tf.greater(x, 0)

    #then do the following three lines in order to convert the final output of this
    #step function into a 1 or -1 (b/c is what we defined T/F to be above in line 4)
    as_float = tf.to_float(is_greater)

    doubled = tf.multiply(as_float, 2)

    return tf.subtract(doubled, 1) 



output = step(tf.matmul(training_input, w))

error = tf.subtract(training_output, output)

mse = tf.reduce_mean(tf.square(error))



delta = tf.matmul(training_input, error, transpose_a=True)

train = tf.assign(w, tf.add(w, delta))



sess = tf.Session()

sess.run(tf.global_variables_initializer())



err, target = 1, 0

epoch, max_epochs = 0, 10



while err > target and epoch < max_epochs:

    epoch += 1

    err, _ = sess.run([mse, train])

print('epoch:', epoch, 'mse:', err)


print(sess.run(w))
