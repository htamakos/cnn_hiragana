import tensorflow as tf
import numpy as np

IMAGE_SIZE = 28
LABEL_NUM = 75

class Dataset(object):
    def __init__(self, images, labels, one_hot=False, dtype=tf.float32):
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

    @property

    def epochs_completed(self):
        return self._epochs_completed

def inference(x_image, data_type, keep_prob, bn=False):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])

        if bn:
            h_conv1 = conv2d(x_image, W_conv1)
            bn1 = tf.layers.batch_normalization(h_conv1)
            h_pool1 = max_pool2x2(tf.nn.relu(bn1))
        else:
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])

        if bn:
            h_conv2 = conv2d(h_pool1, W_conv2)
            bn2 = tf.layers.batch_normalization(h_conv2)
            h_pool2 = max_pool2x2(tf.nn.relu(bn2))
        else:
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool2x2(h_conv2)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        if bn:
            bn3 = tf.layers.batch_normalization(tf.matmul(h_pool2_flat, W_fc1))
            h_fc1 = tf.nn.relu(bn3)
        else:
            b_fc1 = bias_variable([1024])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        kp = tf.cond(tf.equal(data_type, tf.constant(0, tf.int32)), lambda: tf.constant(1.0, tf.float32), lambda: keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1, kp)

    with tf.name_scope('output'):
        W_fc2 = weight_variable([1024, LABEL_NUM])
        b_fc2 = bias_variable([LABEL_NUM])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv

def training(loss, optimizer, global_step, learning_rate):
    return eval('tf.train.' + optimizer + '(learning_rate)') \
               .minimize(loss, global_step=global_step)

def loss(logits, labels):
    #return tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits)))
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def evaluation(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
