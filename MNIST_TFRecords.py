import sys
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# TensorFlow Tutorial 18
# TFRecords & Dataset API
# by Magnus Erik Hvass Pedersen

def convert_image_into_tfrecords(data_set,
                                 out_path):
    # Args:
    # image_paths   List of file-paths for the images.
    # labels        Class-labels for the images.
    # out_path      File-path for the TFRecords output file.
    # Number of images. Used when printing the progress.

    num_examples, rows, cols, depth = data_set.images.shape

    images = data_set.images
    images = images*255.
    images = images.astype(np.uint8)

    labels = data_set.labels

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        # Iterate over all the image-paths and class-labels.
        for i, (img, lbl) in enumerate(zip(images, labels)):
            # Print the percentage-progress.
            print_progress(count=i, total=num_examples-1)

            # Convert the image to raw bytes.
            image_raw = img.tostring()

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(image_raw),
                'label': _int64_feature(int(lbl))
            }))

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)

    print("  Writing {} done!".format(out_path))


class FLAGS(object):

    def __init__(self,
                 path_tfr_train,
                 path_tfr_valid):

        self.path_tfr_train = path_tfr_train
        self.path_tfr_valid = path_tfr_valid


if __name__ == '__main__':
    # Define path to data and hyperparameters for training the model.
    flags = FLAGS(path_tfr_train='C:\\Users\\Diego Lozano\\AFRL_Project\\MNIST_TFRecords\\train_MNIST.tfrecords',
                  path_tfr_valid='C:\\Users\\Diego Lozano\\AFRL_Project\\MNIST_TFRecords\\validation_MNIST.tfrecords')

    mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data",
                                      reshape=False)

    train_dataset = mnist.train
    validation_dataset = mnist.validation

    convert_image_into_tfrecords(train_dataset,
                                 flags.path_tfr_train)

    convert_image_into_tfrecords(validation_dataset,
                                 flags.path_tfr_valid)
