'''
Created on Dec 08, 2016

@author: behrooz
'''
import os
import tensorflow as tf
from sequence_reader import Sequence_Reader
from sequence_model import Sequence_Classifier
from sequence_model import loss
import time
import re
import numpy as np

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)





def tower_loss(scope, FLAGS, sequence_reader, sequence_classifier):


    sequence, labels = sequence_reader.get_next_training_batch(FLAGS)

    # Build inference Graph.
    logits = sequence_classifier.inference(sequence)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = loss(logits, labels, sequence_classifier.get_vars(), FLAGS)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % 'sequence_model_tower', '', l.op.name)
        tf.scalar_summary(loss_name, l)

    return total_loss

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
    
        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)
        
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train(FLAGS):
    
    sequence_reader = Sequence_Reader(FLAGS)
    sequence_classifier = Sequence_Classifier(FLAGS)
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    
    # Adam optimizer for loss
    optimizer = tf.train.AdamOptimizer()
    
    # Calculate the gradients for each model tower.
    tower_grads = []  
    
    for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % ('sequence_model_tower', i)) as scope:
                # Calculate the loss for one tower of the CIFAR model. This function
                # constructs the entire CIFAR model but shares the variables across
                # all towers.
                loss = tower_loss(scope, FLAGS, sequence_reader, sequence_classifier)
                
                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()
                
                # Retain the summaries from the final tower.
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                
                # Calculate the gradients for the batch of data on this CIFAR tower.
                grads = optimizer.compute_gradients(loss)
                
                # Keep track of the gradients across all towers.
                tower_grads.append(grads)
    
    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)
   
    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = optimizer.apply_gradients(grads)
    
    
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)    
    
    
    #####################
    # evaluation
    test_sequence, test_labels = sequence_reader.get_next_test_batch(FLAGS)

    # Build inference Graph.
    test_logits = sequence_classifier.inference(test_sequence)
    test_output = test_logits[-1]
    
    test_op = test_output, test_labels

    #####################
    
    
    for var in tf.all_variables():
        variable_summaries(var, var.name) 
    
    
    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.merge_summary(summaries)

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()    
    

    
    # Build an initialization operation to run below.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    tf.train.start_queue_runners(sess=sess)

    for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss])
        duration = time.time() - start_time
        print(loss_value, duration)
        
        if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
        
        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:            
            # Evaluating on test_iterations_in_one_train_iteration
            test_confusion_matrix = np.zeros((FLAGS.output_size, FLAGS.output_size), dtype=np.int32)
            correctly_classified = 0.0
            total_examples = 0.0
            for test_step in xrange(FLAGS.test_iterations_in_one_train_iteration):
                estimated_labels, ground_truth = sess.run(test_op)
                estimated_labels = np.argmax(estimated_labels, axis=1)
                total_examples += len(ground_truth)
                for j in xrange(len(ground_truth)):
                    if ground_truth[j][-1] == estimated_labels[j]:
                        correctly_classified += 1
                    test_confusion_matrix[ground_truth[j][-1], estimated_labels[j]] += 1
            
            print ('confusion_matrix', test_confusion_matrix)
            print ('accuracy:', correctly_classified / total_examples)
            
            checkpoint_path = os.path.join(FLAGS.model_path, FLAGS.model_prefix + '.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)        
