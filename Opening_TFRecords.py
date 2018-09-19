import tensorflow as tf


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

    # Creating a variable that will normalize the input image
    norm = tf.constant(255, dtype=tf.float32)

    # Decode the raw bytes so it becomes a tensor with type.
    input_image = tf.decode_raw(features['image_raw'], tf.uint8)

    # The type is now uint8 but we need it to be float.
    input_image = tf.cast(input_image, tf.float32)

    # When it is raw the data is a vector and we need to reshape it.
    input_image = tf.reshape(input_image, [28, 28, 1])

    # Normalize the data before feeding it through. Optional.
    input_image = tf.divide(input_image, norm)

    label = features['label']

    return input_image, label


class DatasetTFRecords(object):

    def __init__(self, path_tfrecords_train, path_tfrecords_valid, batch_size):

        train_dataset = tf.data.TFRecordDataset(filenames=path_tfrecords_train)
        # Parse the serialized data in the TFRecords files.
        # This returns TensorFlow tensors for the image and labels.
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
        # and test.
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


class FLAGS(object):

    def __init__(self,
                 batch_size,
                 path_tfr_train,
                 path_tfr_valid):

        self.batch_size = batch_size
        self.path_tfr_train = path_tfr_train
        self.path_tfr_valid = path_tfr_valid


if __name__ == '__main__':

    flags = FLAGS(batch_size=1,
                  path_tfr_train="C:\\Users\\Diego Lozano\\AFRL_Project\\MNIST_TFRecords\\train_MNIST.tfrecords",
                  path_tfr_valid="C:\\Users\\Diego Lozano\\AFRL_Project\\MNIST_TFRecords\\validation_MNIST.tfrecords")

    dataset = DatasetTFRecords(path_tfrecords_train=flags.path_tfr_train,
                               path_tfrecords_valid=flags.path_tfr_valid,
                               batch_size=flags.batch_size)

    y = tf.placeholder(tf.int64, [None])
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])

    with tf.Session() as sess:

        dataset.initialize_training_iterator(sess)

        images, label = dataset.get_next_training_element(sess)

        # Make a dict to load the batch onto the placeholders.
        feed_dict = {x: images,
                     y: label}

        print(sess.run([x, y], feed_dict=feed_dict))