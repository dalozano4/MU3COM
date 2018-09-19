import sys
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO

slim = tf.contrib.slim


class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_image(self, tag, img, step):
        """Original version Logs a list of images."""
        """Updated version logs one image"""

        # Changes that were made were to comment the loop over a list of
        # images.
        # Change the input from images to img since we are passing only one
        # image everytime the function is called

        im_summaries = []

        s = BytesIO()

        plt.imsave(s, img, format='png')

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=img.shape[0],
                                   width=img.shape[1])

        # Create a Summary value
        im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, 0),
                                             image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)

    ###########################################################################
    # def log_images(self, tag, images, step):
    #     """Logs a list of images."""
    #     im_summaries = []
    #     for nr, img in enumerate(images):
    #         # Write the image to a string
    #         # s = StringIO()
    #         s = BytesIO()
    #         plt.imsave(s, img, format='png')

    #         # Create an Image object
    #         img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
    #                                    height=img.shape[0],
    #                                    width=img.shape[1])
    #         # Create a Summary value
    #         im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
    #                                              image=img_sum))

    #     # Create and write Summary
    #     summary = tf.Summary(value=im_summaries)
    #     self.writer.add_summary(summary, step)
    ######################################################################

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])

        self.writer.add_summary(summary, step)

        self.writer.flush()


def _read_and_decode(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    features = \
        {
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

    features = tf.parse_single_example(serialized=serialized,
                                       features=features)

    # Decode the raw bytes so it becomes a tensor with type.
    input_image = tf.decode_raw(features['image_raw'], tf.uint8)

    # Creating a variable that will normalize the input image
    norm = tf.constant(255, dtype=tf.float32)

    # When it is raw the data is a vector and we need to reshape it.
    input_image = tf.reshape(input_image, [28, 28, 1])
    # The type is now uint8 but we need it to be float.
    input_image = tf.cast(input_image, tf.float32)

    # Normalize the data before feeding it through. Optional.
    input_image = tf.divide(input_image, norm)

    label = features['label']

    return input_image, label


class DatasetTFRecords(object):

    def __init__(self, path_tfrecords_train, path_tfrecords_valid, batch_size):

        train_dataset = tf.data.TFRecordDataset(filenames=path_tfrecords_train)
        # Parse the serialized data in the TFRecords files.
        # This returns TensorFlow tensors for the input image and target image.
        train_dataset = train_dataset.map(_read_and_decode)

        # String together various operations to apply to the data
        train_dataset = train_dataset.shuffle(1000)
        train_dataset = train_dataset.batch(batch_size)
        # Will iterate through the data once before throwing an OutOfRangeError
        self._train_iterator = train_dataset.make_one_shot_iterator()

        self._train_init_op = self._train_iterator.make_initializer(train_dataset)

        validation_dataset = tf.data.TFRecordDataset(filenames=path_tfrecords_valid)

        validation_dataset = validation_dataset.map(_read_and_decode)

        validation_dataset = validation_dataset.shuffle(1000)
        validation_dataset = validation_dataset.batch(batch_size)
        self._validation_iterator = validation_dataset.make_one_shot_iterator()

        self._validation_init_op = self._validation_iterator.make_initializer(validation_dataset)

        # Create a placeholder that can be dynamically changed between train
        # and validation.
        self._handle = tf.placeholder(tf.string, shape=[])

        # Define a generic iterator using the shape of the dataset
        iterator = tf.data.Iterator.from_string_handle(self._handle,
                                                       train_dataset.output_types,
                                                       train_dataset.output_shapes)

        self._next_element = iterator.get_next()
        self._train_handle = []
        self._validation_handle = []

    def initialize_training_iterator(self, sess):
        sess.run(self._train_init_op)

    def initialize_validation_iterator(self, sess):
        sess.run(self._validation_init_op)

    def get_next_training_element(self, sess):
        # The `Iterator.string_handle()` method returns a tensor that can be
        # evaluated and used to feed the `handle` placeholder.
        self._train_handle = sess.run(self._train_iterator.string_handle())
        feed_dict = {self._handle: self._train_handle}
        elements = sess.run(self._next_element, feed_dict=feed_dict)
        return elements

    def get_next_validation_element(self, sess):
        self._validation_handle = sess.run(self._validation_iterator.string_handle())
        feed_dict = {self._handle: self._validation_handle}
        elements = sess.run(self._next_element, feed_dict=feed_dict)
        return elements


class Model(object):
    def __init__(self,
                 img_size=28,
                 num_channels=1):

        self.num_channels = num_channels

        self.x_placeholder = tf.placeholder(tf.float32, [None, img_size, img_size, num_channels])

        self.y_placeholder = tf.placeholder(tf.int64, [None])

    def mnist_model(self):

        conv_1 = slim.conv2d(self.x_placeholder, 32, [5, 5], stride=1, scope='conv1')
        conv_1 = slim.max_pool2d(conv_1, [2, 2], scope='pool1')

        conv_2 = slim.conv2d(conv_1, 64, [5, 5], stride=1, scope='conv2')
        conv_2 = slim.max_pool2d(conv_2, [2, 2], scope='pool2')

        flattened = tf.reshape(conv_2, [-1, 7 * 7 * 64])

        dense_layer_1 = slim.fully_connected(flattened, 1024, scope='fc_1')
        dropout = slim.dropout(dense_layer_1, keep_prob=0.4)

        dense_layer_2 = slim.fully_connected(dropout, 10, activation_fn=None, scope='fc_2')

        return tf.nn.sigmoid(dense_layer_2), dense_layer_2


def log_weights_bias():
    for variable in slim.get_model_variables():
        with tf.name_scope(variable.op.name):
            tf.summary.scalar('mean', tf.reduce_mean(variable))

            tf.summary.scalar('max', tf.reduce_max(variable))

            tf.summary.scalar('min', tf.reduce_min(variable))

            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - tf.reduce_mean(variable))))

            tf.summary.scalar('stddev', stddev)

            tf.summary.histogram('histogram', variable)


def training_model(_):

    # Clear the log directory, if it exists.
    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    # Create a log directory, if it exists.
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Reset the default graph.
    tf.reset_default_graph()

    model = Model()

    dataset = DatasetTFRecords(path_tfrecords_train=FLAGS.path_tfrecords_train,
                               path_tfrecords_valid=FLAGS.path_tfrecords_valid,
                               batch_size=FLAGS.batch_size)

    logger = Logger(FLAGS.log_dir)

    true_value = tf.one_hot(model.y_placeholder, 10)

    model_output_tensor, model_logit = model.mnist_model()

    # Now, we construct a loss function.
    with tf.name_scope("loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_value, logits=model_logit)
        loss = tf.reduce_mean(cross_entropy)

    # Next, add an optimizer to the graph.
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

    # Set up the accuracy
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(true_value, 1), tf.argmax(model_output_tensor, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    log_weights_bias()

    summary = tf.summary.merge_all()

    sv = tf.train.Supervisor(logdir=FLAGS.log_dir,
                             summary_op=None)

    with sv.managed_session() as sess:

        for i in range(FLAGS.epochs):

            dataset.initialize_training_iterator(sess)
            print("<------------Training_output------------->")

            # Train for one epoch.
            while True:

                try:

                    start_time = time.time()

                    images, label = dataset.get_next_training_element(sess)

                    # Make a dict to load the batch onto the placeholders.
                    feed_dict = {model.x_placeholder: images,
                                 model.y_placeholder: label}

                    # Not a train loss.
                    train_loss, train_accuracy = sess.run([loss, accuracy], feed_dict=feed_dict)

                    sess.run(optimizer, feed_dict=feed_dict)

                    print("Running time = " + str(time.time() - start_time))

                except tf.errors.OutOfRangeError:

                    break

            print("Training Epoch: {}, Train_Loss: {:.3f}, Train_Acc: {:.3f}".format(i, train_loss, train_accuracy))

            print("<-------------Saving Training Variables-------------->")

            logger.log_scalar('train_loss', train_loss, i)
            logger.log_scalar('train_accuracy', train_accuracy, i)

            dataset.initialize_validation_iterator(sess)

            print("<------------Validation_output------------->")
            while True:

                    try:

                        images, label = dataset.get_next_validation_element(sess)

                        # Make a dict to load the batch onto the placeholders.
                        feed_dict = {model.x_placeholder: images,
                                     model.y_placeholder: label}

                        valid_loss, valid_accuracy = sess.run([loss, accuracy], feed_dict=feed_dict)

                    except tf.errors.OutOfRangeError:

                        break

            print("Training Epoch: {}, Valid_Loss: {:.3f}, Valid_Acc: {:.3f}".format(i, valid_loss, valid_accuracy))

            print("<-------------Saving Validation Variables-------------->")

            logger.log_scalar('validation_loss', valid_loss, i)
            logger.log_scalar('validation_accuracy', valid_accuracy, i)

            logger.log_image('validation_image', images[0, :, :, 0], i)
            print("<------------- Saving Weights and Bias -------------->")

            summary_str = sess.run(summary, feed_dict=feed_dict)
            logger.writer.add_summary(summary_str, i)

        sv.stop()
        sess.close()


if __name__ == '__main__':
    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    # Establish default arguements.

    # These flags are often, but not always, overwritten by the launcher.
    parser.add_argument('--path_tfrecords_train', type=str,
                        default="C:\\Users\\Diego Lozano\\AFRL_Project\\MNIST_TFRecords\\train_MNIST.tfrecords",
                        help='Location of the training data set which is in .tfrecords format.')

    parser.add_argument('--path_tfrecords_valid', type=str,
                        default="C:\\Users\\Diego Lozano\\AFRL_Project\\MNIST_TFRecords\\validation_MNIST.tfrecords",
                        help='Location of the test data set which is in .tfrecords format.')

    parser.add_argument('--log_dir', type=str,
                        default='C:\\Users\\Diego Lozano\\AFRL_Project\\MNISTlog',
                        help='Summaries log directory.')

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate.')

    parser.add_argument('--img_size', type=int, default=28,
                        help='The image size.')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training set batch size.')

    parser.add_argument('--epochs', type=int, default=8,
                        help='Number of epochs to run trainer.')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=training_model, argv=[sys.argv[0]] + unparsed)

    # tensorboard --logdir="C:\\Users\\Diego Lozano\\AFRL_Project\\MNISTlog"
