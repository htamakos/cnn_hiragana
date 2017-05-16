import time
import os
import sys
import argparse

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import scipy.misc
import scipy.ndimage

DATA_DIR = '../data/ETL8G'
HIRAGANA_DATA_DIR = DATA_DIR + '/hiragana_images/'
LABELS = ['0x2422', '0x2424', '0x2426', '0x2428', '0x242a', '0x242b',
          '0x242c', '0x242d', '0x242e', '0x242f', '0x2430', '0x2431',
          '0x2432', '0x2433', '0x2434', '0x2435', '0x2436', '0x2437',
          '0x2438', '0x2439', '0x243a', '0x243b', '0x243c', '0x243d',
          '0x243e', '0x243f', '0x2440', '0x2441', '0x2442', '0x2443',
          '0x2444', '0x2445', '0x2446', '0x2447', '0x2448', '0x2449',
          '0x244a', '0x244b', '0x244c', '0x244d', '0x244e', '0x244f',
          '0x2450', '0x2451', '0x2452', '0x2453', '0x2454', '0x2455',
          '0x2456', '0x2457', '0x2458', '0x2459', '0x245a', '0x245b',
          '0x245c', '0x245d', '0x245e', '0x245f', '0x2460', '0x2461',
          '0x2462', '0x2463', '0x2464', '0x2465', '0x2466', '0x2467',
          '0x2468', '0x2469', '0x246a', '0x246b', '0x246c', '0x246d',
          '0x246f', '0x2472', '0x2473']
NPZ = 'hiragana.npz'
IMAGE_SIZE = 28
LABEL_NUM = 75

DECAY_RATE = 0.96
DECAY_RATE_KP = 0.8
DECAY_STEPS = 5000

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

def prepare_data():
    dataset = np.load(NPZ)
    images = dataset['image']
    labels = dataset['label']

    # imagesに入っているデータは0-15のグレースケール
    # 15で割ることによって正規化を行っている
    images = images.reshape([-1, 127, 128]).astype(np.float32) / 15

    X = np.array([])
    Y = np.array([])

    if FLAGS.data_argument == 'true':
        X = np.zeros([images.shape[0] * 3, IMAGE_SIZE, IMAGE_SIZE])
        Y = np.zeros([labels.shape[0] * 3, LABEL_NUM])
        for i in range(images.shape[0]):
            img = scipy.misc.imresize(images[i], (IMAGE_SIZE, IMAGE_SIZE), mode='F')
            label = labels[i]
            index = i * 3
            X[index] = img
            X[index+1] = scipy.ndimage.interpolation.rotate(img, +15, reshape=False)
            Y[index+1] = label
            X[index+2] = scipy.ndimage.interpolation.rotate(img, -15, reshape=False)
            Y[index+2] = label
    else:
        X = np.zeros([images.shape[0], IMAGE_SIZE, IMAGE_SIZE])
        Y = np.zeros([labels.shape[0], LABEL_NUM])
        for i in range(images.shape[0]):
            img = scipy.misc.imresize(images[i], (IMAGE_SIZE, IMAGE_SIZE), mode='F')
            label = labels[i]
            X[i] = img
            Y[i] = label

    return train_test_split(X, Y, random_state=0, test_size=FLAGS.test_size)

def print_stdout(s):
    print(s)
    sys.stdout.flush()

def train():
    def feed_dict(_x, _y, _data_type):
        return { x: _x, y_: _y, data_type: _data_type }

    train_images, test_images, train_labels, test_labels = prepare_data()
    etl8g = ETL8G(train_images, train_labels)

    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE])
        y_ = tf.placeholder(tf.float32, shape=[None, LABEL_NUM])
        x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        tf.summary.image('image', x_image, 100)

        global_step = tf.Variable(0, trainable=False)

        if FLAGS.optimizer == 'AdamOptimizer':
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, DECAY_STEPS, DECAY_RATE, staircase=True)
        elif FLAGS.optimizer in 'AdagradOptimizer':
            learning_rate = tf.constant(0.01)
        elif FLAGS.optimizer == 'AdadeltaOptimizer':
            learning_rate = tf.constant(0.001)

        keep_prob = tf.train.exponential_decay(FLAGS.train_keep_prob, global_step, DECAY_STEPS, DECAY_RATE_KP, staircase=True)
        data_type = tf.placeholder(tf.int32)

        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 1, 32])

            if FLAGS.bn:
                h_conv1 = conv2d(x_image, W_conv1)
                bn1 = tf.layers.batch_normalization(h_conv1)
                h_pool1 = max_pool2x2(tf.nn.relu(bn1))
            else:
                b_conv1 = bias_variable([32])
                h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
                h_pool1 = max_pool2x2(h_conv1)

        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 32, 64])

            if FLAGS.bn:
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

            if FLAGS.bn:
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

        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('keep_prob', keep_prob)

        #loss = -tf.reduce_sum(y_ * tf.log(y_conv))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
        tf.summary.scalar('loss', loss)

        train_step = eval('tf.train.' + FLAGS.optimizer + '(learning_rate)') \
                         .minimize(loss, global_step=global_step)

        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)
            sess.run(init)

            accracies = []
            batch_counter = 0

            start_time = time.time()

            print_stdout(("train_data_num %d, test_data_num %d batch_size %d max_epoch %d " + \
                         "start_learning_rate %0.5f optimizer %s train_keep_prob %0.5f " + \
                         "batch normalization %r") % \
                         (train_images.shape[0], test_images.shape[0], FLAGS.batch_num, FLAGS.max_epoch,
                          FLAGS.learning_rate, FLAGS.optimizer, FLAGS.train_keep_prob, FLAGS.bn))

            epoc_count = 0

            while epoc_count <=  FLAGS.max_epoch:
                batch_x, batch_y = etl8g.next_batch(FLAGS.batch_num)

                sess.run(train_step, feed_dict=feed_dict(batch_x, batch_y, 1))
                epoch = etl8g.epochs_completed

                if epoc_count != epoch:
                    duration = time.time() - start_time

                    train_accuracy = accuracy.eval(feed_dict=feed_dict(batch_x, batch_y, 0))
                    accracies.append(train_accuracy)
                    print_stdout("epoch %d, num_example %d, training accuracy %.03f (%.03f sec)" % \
                                 (epoch, batch_x.shape[0], train_accuracy, duration))

                    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(batch_x, batch_y, 1))
                    train_writer.add_summary(summary, epoch)

                    test_feed_dict = feed_dict(test_images, test_labels, 0)
                    test_accuracy = accuracy.eval(feed_dict=test_feed_dict)
                    test_summary = sess.run(merged, feed_dict=test_feed_dict)

                    test_writer.add_summary(test_summary, epoch)
                    print_stdout("num_example %d, test accuracy %.03f" % (test_images.shape[0], test_accuracy))
                    epoc_count += 1

            duration = time.time() - start_time

            print_stdout("num_example %d, test accuracy %.03f (%.03f sec)" % (test_images.shape[0], accuracy.eval(
                         feed_dict(test_images, test_labels, 0)), duration))

            print_stdout("epoch %d, train_data_size %d, test_data_size %d batch_size %d max_step %d" % \
                         (etl8g.epochs_completed, train_images.shape[0], test_images.shape[0],
                          FLAGS.batch_num, FLAGS.batch_size))

def main(_):
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=400,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_num', type=int, default=50,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--log_dir', type=str, default='./cnn_hiragana_logs',
                        help='Summaries log directory')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test size')
    parser.add_argument('--train_keep_prob', type=float, default=0.5,
                        help='Keep rate in Dropout layer')
    parser.add_argument('--optimizer', type=str, default='AdamOptimizer',
                        help='Loss Optimizer in DeepLearning')
    parser.add_argument('--bn', type=bool, default=True,
                        help='Batch Normalization')
    parser.add_argument('--data_argument', type=str, default='false',
                        help='Data Argumentation')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
