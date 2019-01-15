import tensorflow as tf 


def step(x):



    is_greater = tf.greater(x, 0)

    #then do the following three lines in order to convert the final output of this
    #step function into a 1 or -1 (b/c is what we defined T/F to be above in line 4)
    as_float = tf.to_float(is_greater)

    doubled = tf.multiply(as_float, 2)

    return tf.subtract(doubled, 1)