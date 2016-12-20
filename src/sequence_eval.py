'''
Created on Oct, 2016

@author: mitra
'''
import os
import tensorflow as tf
from sequence_reader import Sequence_Reader
from sequence_model import Sequence_Classifier
import time
import re
import numpy as np


def eval(FLAGS):
    
    sequence_reader = Sequence_Reader(FLAGS)
    sequence_classifier = Sequence_Classifier(FLAGS)

    
    sequence, labels = sequence_reader.get_next_test_batch(FLAGS)

    # Build inference Graph.
    logits = sequence_classifier.inference(sequence)
    output = logits[-1]
    
    test_op = output, labels
    
        
    sess = tf.Session()


    saver = tf.train.Saver()
    tf.train.start_queue_runners(sess=sess)

    confusion_matrix = np.zeros((FLAGS.output_size, FLAGS.output_size), dtype=np.int32)
    with sess:
        saver.restore(sess, FLAGS.model_path)
        for i in xrange(1000):
            estimated_labels, ground_truth = sess.run(test_op)
            estimated_labels = np.argmax(estimated_labels, axis=1)
            local_confusion_matrix = np.zeros((FLAGS.output_size, FLAGS.output_size), dtype=np.int32)
            ground_truth = np.squeeze(ground_truth)
            
            for j in xrange(len(ground_truth)):                
                confusion_matrix[ground_truth[j][-1], estimated_labels[j]] += 1
                local_confusion_matrix[ground_truth[j], estimated_labels[j]] += 1            
            # print('local_confusion_matrix', local_confusion_matrix)
    print('confusion_matrix', confusion_matrix)
            
