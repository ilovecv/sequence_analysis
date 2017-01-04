'''
Created on Oct, 2016

@author: behrooz
'''
import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np

def _variable_on_cpu(name, shape, initializer, FLAGS):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = FLAGS.input_format
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, FLAGS):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = FLAGS.input_format
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=FLAGS.stddev, dtype=dtype), FLAGS)
    weight_decay = tf.mul(tf.nn.l2_loss(var), FLAGS.weight_decay, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return var

def l2_regularization(variables, weight_decay):
    l2_loss = 0
    for var in variables:
        l2_loss += weight_decay * tf.nn.l2_loss(var)
    return l2_loss


def loss(estimated_labels, ground_truth, variables, FLAGS):
    
    ground_truth = tf.transpose(ground_truth, [1, 0, 2])  # permute sequence_size and batch_size
    ground_truth = tf.unpack(ground_truth)

    
    logits = estimated_labels[-1]
    labels = tf.squeeze(ground_truth[-1])
    ratio = 0.2
    class_weight = tf.constant([ratio, 1.0 - ratio])
    weighted_logits = tf.mul(logits, class_weight)  # shape [batch_size, 2]

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(weighted_logits, labels)
    
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    
    tf.add_to_collection('losses', cross_entropy_mean)
    
    loss = cross_entropy_mean + l2_regularization(variables, FLAGS.weight_decay)
    
    return tf.add_n(tf.get_collection('losses'), name='total_loss')



class Sequence_Classifier():

    def __init__(self, FLAGS):
        """Initialization.

        Args:
            FLAGS: flags object
        """
        self.flags = FLAGS
        self.number_layers = FLAGS.number_layers
        self.number_hidden = FLAGS.number_hidden
        self.input_size = FLAGS.input_size
        self.batch_size = FLAGS.batch_size
        self.sequence_size = FLAGS.sequence_size
        self.output_size = FLAGS.output_size
        self.name = "sequence_model"
        print 'Initializing Sequence Original Classifier'
        print 'input size:', FLAGS.input_size
        print 'batch size:', FLAGS.batch_size
        print 'sequence size:', FLAGS.sequence_size
        print 'hidden size:', FLAGS.number_hidden
        
    def inference(self, input_sequence, reuse=False):      
        with tf.variable_scope(self.name):
            if reuse == True:
                tf.get_variable_scope().reuse_variables()
            
#             beta = tf.Variable(tf.constant(0.0, shape=[self.flags.input_size]), name='beta', trainable=True)
#             gamma = tf.Variable(tf.constant(1.0, shape=[self.flags.input_size]), name='gamma', trainable=True)
            
            batch_mean1, batch_var1 = tf.nn.moments(input_sequence, [0])

            input_sequence_bn = tf.nn.batch_normalization(input_sequence, batch_mean1, batch_var1, None, None, 0.00001)
                
            # Define weights
#             sequence_classifier_weights = tf.Variable(tf.random_normal([self.number_hidden, self.output_size]), name='output_weights')
            sequence_classifier_weights = _variable_with_weight_decay(name='output_weights', shape=[self.number_hidden, self.output_size], FLAGS=self.flags)
#             sequence_classifier_biases = tf.Variable(tf.random_normal([self.output_size]), name="output_biases")
            sequence_classifier_biases = _variable_with_weight_decay(name='output_biases', shape=[self.output_size], FLAGS=self.flags)
            
            #
            sequence_classifier_input = tf.reshape(input_sequence_bn, [self.batch_size, self.sequence_size, self.input_size]) 

            # input_sequence shape: (batch_size, n_steps, input_size)    
            sequence_classifier_input = tf.transpose(sequence_classifier_input, [1, 0, 2])  # permute sequence_size and batch_size
            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            sequence_classifier_input = tf.unpack(sequence_classifier_input) 

            sequence_classifier_recurrent_cell = tf.nn.rnn_cell.LSTMCell(self.number_hidden, forget_bias=1.0, state_is_tuple=True)
            
            if (self.flags.dropout == True):
                print ('adding dropout layer to LSTM')
                # adding dropout to the input_sequence-hidden layer weights
                sequence_classifier_recurrent_cell = tf.nn.rnn_cell.DropoutWrapper(sequence_classifier_recurrent_cell, output_keep_prob=self.flags.keep_prob)

            # creating stack of several hidden units
            sequence_classifier_stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([sequence_classifier_recurrent_cell] * self.number_layers, state_is_tuple=True)
            
            sequence_classifier_hidden_outputs, hidden_states = tf.nn.rnn(sequence_classifier_stacked_lstm, sequence_classifier_input, dtype=self.flags.input_format)
                        
            sequence_classifier_output = [tf.add(tf.matmul(h, sequence_classifier_weights), sequence_classifier_biases) for h in sequence_classifier_hidden_outputs]
            
            return sequence_classifier_output
        
    def get_vars(self):
        return [var for var in tf.trainable_variables() if (self.name + "/") in var.name]


class Sequence_Classifier_With_Convolution():

    def __init__(self, FLAGS):
        """Initialization.

        Args:
            FLAGS: flags object
        """
        self.flags = FLAGS
        self.number_layers = FLAGS.number_layers
        self.number_hidden = FLAGS.number_hidden
        self.input_size = FLAGS.input_size
        self.batch_size = FLAGS.batch_size
        self.sequence_size = FLAGS.sequence_size
        self.output_size = FLAGS.output_size
        self.feature_map_size = FLAGS.feature_map_size
        self.name = "sequence_model"
        self.conv_size = FLAGS.conv_size
        self.conv_stride = FLAGS.conv_stride
        print 'Initializing Convolutional Sequence Classifier'
        print 'input size:', FLAGS.input_size
        print 'batch size:', FLAGS.batch_size
        print 'sequence size:', FLAGS.sequence_size
        print 'feature map size:', FLAGS.feature_map_size
        print 'hidden size:', FLAGS.number_hidden
        print 'convolution size:', FLAGS.conv_size
        print 'convolution stride:', FLAGS.conv_stride

    def inference(self, input_sequence_orig, reuse=False):
        with tf.variable_scope(self.name):
            batch_mean1, batch_var1 = tf.nn.moments(input_sequence_orig, [0])

            input_sequence_bn = tf.nn.batch_normalization(input_sequence_orig, batch_mean1, batch_var1, None, None, 0.00001)
            # Run the 1-D conv
            filter = _variable_with_weight_decay('kernel_weights', shape=[self.conv_size, self.input_size, self.feature_map_size], FLAGS=self.flags)
            input_sequence = tf.nn.conv1d(input_sequence_bn, filter, stride=self.conv_stride, padding="SAME")

            print 'after applying cnn', input_sequence.get_shape()

            # Define weights
#             sequence_classifier_weights = tf.Variable(tf.random_normal([self.number_hidden, self.output_size]), name='output_weights')
            sequence_classifier_weights = _variable_with_weight_decay(name='output_weights', shape=[self.number_hidden, self.output_size], FLAGS=self.flags)
#             sequence_classifier_biases = tf.Variable(tf.random_normal([self.output_size]), name="output_biases")
            sequence_classifier_biases = _variable_with_weight_decay(name='output_biases', shape=[self.output_size], FLAGS=self.flags)
            
            #
            sequence_classifier_input = tf.reshape(input_sequence, [self.batch_size, input_sequence.get_shape()[1], self.feature_map_size]) 

            # input_sequence shape: (batch_size, n_steps, input_size)    
            sequence_classifier_input = tf.transpose(sequence_classifier_input, [1, 0, 2])  # permute sequence_size and batch_size
            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            sequence_classifier_input = tf.unpack(sequence_classifier_input) 

            sequence_classifier_recurrent_cell = tf.nn.rnn_cell.LSTMCell(self.number_hidden, forget_bias=1.0, state_is_tuple=True)
            
            if (self.flags.dropout == True):
                print ('adding dropout layer to LSTM')
                # adding dropout to the input_sequence-hidden layer weights
                sequence_classifier_recurrent_cell = tf.nn.rnn_cell.DropoutWrapper(sequence_classifier_recurrent_cell, output_keep_prob=self.flags.keep_prob)

            # creating stack of several hidden units
            sequence_classifier_stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([sequence_classifier_recurrent_cell] * self.number_layers, state_is_tuple=True)
            
            sequence_classifier_hidden_outputs, hidden_states = tf.nn.rnn(sequence_classifier_stacked_lstm, sequence_classifier_input, dtype=self.flags.input_format)
                        
            sequence_classifier_output = [tf.add(tf.matmul(h, sequence_classifier_weights), sequence_classifier_biases) for h in sequence_classifier_hidden_outputs]
            
            return sequence_classifier_output
        
    def get_vars(self):
        return [var for var in tf.trainable_variables() if (self.name + "/") in var.name]
