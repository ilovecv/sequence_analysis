'''
Created on Dec 08, 2016

@author: behrooz
'''
import os
import tensorflow as tf

class Sequence_Reader():
    
    def __init__(self, FLAGS=None):
        if FLAGS != None: 
            self.train_file_name_prefix = FLAGS.train_file_name_prefix
            self.train_file_name_postfix = FLAGS.train_file_name_postfix
            self.test_file_name_prefix = FLAGS.test_file_name_prefix
            self.test_file_name_postfix = FLAGS.test_file_name_postfix
        else:      
            self.train_file_name_prefix = "seq_"
            self.train_file_name_postfix = ".bin"
            self.test_file_name_prefix = "seq_"
            self.test_file_name_postfix = ".bin"        
    def read_sequence(self, FLAGS, filename_queue):

        class Sequence(object):
            pass
        result = Sequence()
        _dtype = FLAGS.input_format
        if FLAGS.input_format == tf.float32:
            num_bits = 4
        else:
            num_bits = 8
        label_bytes = num_bits  # we assume the entire binary file is recored in FLAGS.input_format format.
               
        sequence_bytes = num_bits * FLAGS.input_size  # we assume the entire binary file is recored in FLAGS.input_format format.
        
        # Every record consists of a sequence followed by a label
        
        record_bytes = FLAGS.sequence_size * (sequence_bytes + label_bytes)
        
        # Read a record, getting filenames from the filename_queue. We assume there is not header or footer bytes
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        result.key, value = reader.read(filename_queue)
        
        # Convert from a string to a vector of uint8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, _dtype)
        
        record_bytes_reshaped = tf.reshape(record_bytes, [FLAGS.sequence_size, FLAGS.input_size + 1])
        
        
        # The bytes before the label bytes represent the sequence, which we reshape
        # from [FLAGS.sequence_size * FLAGS.input_size] to [FLAGS.sequence_size, FLAGS.input_size].
        
        result.sequence = tf.slice(record_bytes_reshaped, [0, 0], [FLAGS.sequence_size, FLAGS.input_size])
        
        # The last bytes represent the label, which we convert from uint8->int32.
        result.label = tf.cast(tf.slice(record_bytes_reshaped, [0, FLAGS.input_size], [FLAGS.sequence_size, 1]), tf.int32)
        
        
        
        return result
    
    
    def _generate_sequqnce_and_label_batch(self, sequence, label, min_queue_examples,
                                        FLAGS):
        num_preprocess_threads = 16
        if FLAGS.shuffle:
            sequences, label_batch = tf.train.shuffle_batch(
                [sequence, label],
                batch_size=FLAGS.batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + FLAGS.batch_size,
                min_after_dequeue=min_queue_examples)
        else:
            sequences, label_batch = tf.train.batch(
                  [sequence, label],
                  batch_size=FLAGS.batch_size,
                  num_threads=num_preprocess_threads,
                  capacity=min_queue_examples + FLAGS.batch_size)
                
        return sequences, label_batch
        
    
    
    
    def get_next_training_batch(self, FLAGS):
    
        """Construct input for training using the Reader ops."""
        
        filenames = [os.path.join(FLAGS.data_dir, self.train_file_name_prefix + '%d' % i + self.train_file_name_postfix) for i in xrange(FLAGS.train_file_range[0], FLAGS.train_file_range[1])] 
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)
        
        # Read examples from files in the filename queue.
        read_input = self.read_sequence(FLAGS, filename_queue)
        
        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 
                                 min_fraction_of_examples_in_queue)
        print ('Filling queue with %d sequences before starting to train. '
               'This will take a few minutes.' % min_queue_examples)
        
        # Generate a batch of sequences and labels by building up a queue of examples.
        return self._generate_sequqnce_and_label_batch(read_input.sequence, read_input.label,
                                               min_queue_examples, FLAGS)



    def get_next_test_batch(self, FLAGS):
    
        """Construct input for test using the Reader ops."""
        
        test_filenames = [os.path.join(FLAGS.data_dir, self.test_file_name_prefix + '%d' % i + self.test_file_name_postfix) for i in xrange(FLAGS.test_file_range[0], FLAGS.train_file_range[1])] 
        for f in test_filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        test_filename_queue = tf.train.string_input_producer(test_filenames)
        
        # Read examples from files in the filename queue.
        test_read_input = self.read_sequence(FLAGS, test_filename_queue)
        
        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 
                                 min_fraction_of_examples_in_queue)
        print ('Filling queue with %d sequences before starting to train. '
               'This will take a few minutes.' % min_queue_examples)
        
        # Generate a batch of sequences and labels by building up a queue of examples.
        return self._generate_sequqnce_and_label_batch(test_read_input.sequence, test_read_input.label,
                                               min_queue_examples, FLAGS)
