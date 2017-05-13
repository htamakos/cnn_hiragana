import time
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_DIR = '../data/ETL8G'
HIRAGANA_DATA_DIR = DATA_DIR + '/hiragana_images/'
LABELS = os.listdir(HIRAGANA_DATA_DIR)
NPZ = '../data/ETL8G/np_hiragana.npz'
MAX_STEP = 200000
BATCH_NUM = 75
LABEL_NUM = 75
LEARNING_RATE = 1e-4
TRAIN_KEEP_PROB = 0.5
TEST_KEEP_PROB = 1.0
IMAGE_SIZE = 28
DECAY_RATE = 0.96

dataset = np.load(NPZ)
images = dataset['image']
labels = dataset['label']

class ETL8G(object):
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

def train():
    train_images, test_images, train_labels, test_labels = \
         train_test_split(images, labels, random_state=0, test_size=0.2)

    etl8g = ETL8G(train_images, train_labels)

    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE])
        y_ = tf.placeholder(tf.float32, shape=[None, LABEL_NUM])
        x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE,1])
        tf.summary.image('image', x_image, 100)

        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

            h_pool1 = max_pool2x2(h_conv1)

        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

            h_pool2 = max_pool2x2(h_conv2)

        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            b_fc1 = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope('output'):
            W_fc2 = weight_variable([1024, LABEL_NUM])
            b_fc2 = bias_variable([LABEL_NUM])
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, MAX_STEP, DECAY_RATE, staircase=True)

        #loss = -tf.reduce_sum(y_ * tf.log(y_conv))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
        tf.summary.scalar('loss', loss)

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            writer = tf.summary.FileWriter("logs", sess.graph)
            sess.run(init)

            accracies = []
            batch_counter = 0

            start_time = time.time()

            for i in range(MAX_STEP):
                batch_x, batch_y = etl8g.next_batch(BATCH_NUM)

                if i % 100 == 0:
                    duration = time.time() - start_time
                    train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_: batch_y, keep_prob: TEST_KEEP_PROB})
                    accracies.append(train_accuracy)
                    print("step %d, num_example %d, training accuracy %.03f (%.03f sec)" % (i, batch_x.shape[0],
                                                                                           train_accuracy, duration))
                    sys.stdout.flush()
                summary, _ = sess.run([merged, train_step], feed_dict={x: batch_x, y_: batch_y, keep_prob: TRAIN_KEEP_PROB})
                writer.add_summary(summary, i)

                if i % 1000 == 0:
                    print("num_example %d, test accuracy %.03f" % (test_images.shape[0], accuracy.eval(
                        feed_dict={x: test_images, y_: test_labels, keep_prob: TEST_KEEP_PROB})))
                    sys.stdout.flush()

            duration = time.time() - start_time
            print("num_example %d, test accuracy %.03f (%.03f sec)" % (test_images.shape[0], accuracy.eval(
                    feed_dict={x: test_images, y_: test_labels, keep_prob: TEST_KEEP_PROB}), duration))
            sys.stdout.flush()

            print("epoch %d, train_data_size %d, test_data_size %d batch_size %d max_step %d" % \
                 (etl8g.epochs_completed, train_images.shape[0], test_images.shape[0], BATCH_NUM, MAX_STEP))

def main():
    train()

if __name__ == '__main__':
    main()
