'''
Created on Oct, 2016

@author: mitra
'''
import itertools
import os
import tensorflow as tf
from sequence_reader import Sequence_Reader
from sequence_model import Sequence_Classifier
from sequence_model import Sequence_Classifier_With_Convolution
import time
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as compute_confusion_matrix


def eval(FLAGS):
    classes_file = open('ucf_train_test_files/classInd.txt')
    classes = []
    for line in classes_file:
        s = line.split(' ')
        classes.append(s.strip)
    sequence_reader = Sequence_Reader(FLAGS)
    if FLAGS.sequence_classifer_type == 1:
        sequence_classifier = Sequence_Classifier(FLAGS)
    else:
        sequence_classifier = Sequence_Classifier_With_Convolution(FLAGS)
    
    sequence, labels = sequence_reader.get_next_test_batch(FLAGS)

    # Build inference Graph.
    logits = sequence_classifier.inference(sequence)
    output = logits[-1]
    
    test_op = output, labels
    
        
    sess = tf.Session()


    saver = tf.train.Saver()
    tf.train.start_queue_runners(sess=sess)

    confusion_matrix = np.zeros((FLAGS.output_size, FLAGS.output_size), dtype=np.int32)
    entire_gt = []
    entire_es = []
    total = 0.0
    correct = 0.0
    with sess:
        saver.restore(sess, FLAGS.model_path)
        for i in xrange(FLAGS.number_of_test_examples):
            estimated_labels, ground_truth = sess.run(test_op)
            estimated_labels = np.argmax(estimated_labels, axis=1)
            local_confusion_matrix = np.zeros((FLAGS.output_size, FLAGS.output_size), dtype=np.int32)
            ground_truth = np.squeeze(ground_truth)
            for j in xrange(len(ground_truth)):                
                confusion_matrix[ground_truth[j][-1], estimated_labels[j]] += 1
                entire_gt.append(ground_truth[j][-1])
                entire_es.append(estimated_labels[j])
                if ground_truth[j][-1] == estimated_labels[j]:
                    correct += 1
            total += len(ground_truth)    
    print('confusion_matrix', confusion_matrix)
    print ('accuracy:' + str(correct / total))
    entire_gt = np.asanyarray(entire_gt, np.int32)
    entire_es = np.asanyarray(entire_es, np.int32)    
