import tensorflow as tf


def conv2d(inputs, num_outputs, kernel_size, scope):
    outputs = tf.contrib.layers.conv2d(
        inputs, num_outputs, kernel_size, scope=scope,
        activation_fn=tf.nn.relu, biases_initializer=None)
    outputs = batch_norm(outputs, scope)
    return outputs

def dropout(inputs, keep_prob=1,scope):
    if keep_prob < 1:
        return tf.contrib.layers.dropout(inputs,keep_prob=keep_prob,scope =scope)
    else:
        return inputs

def pool2d(inputs, kernel_size, scope):
    return tf.contrib.layers.max_pool2d(
        inputs, kernel_size, scope=scope, padding='SAME')


def dense(inputs, dim, scope,activation_fn=tf.nn.relu):
    outputs = tf.contrib.layers.fully_connected(
        inputs, dim, scope=scope+'/dense',activation_fn=activation_fn)
    outputs = batch_norm(outputs, scope)
    return outputs


def batch_norm(inputs, scope):
    return tf.contrib.layers.batch_norm(
        inputs, decay=0.99872, center=True, scale=True, activation_fn=None,
        updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm')
