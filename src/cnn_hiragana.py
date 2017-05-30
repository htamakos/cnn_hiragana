import time
import os
import sys
import argparse

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import scipy.misc
import scipy.ndimage

import etl8g

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
IMAGE_SIZE = 64
LABEL_NUM = 75

DECAY_RATE = 0.1
DECAY_STEPS = 20

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
    def feed_dict(_x, _y, _keep_prob):
        return { x: _x, y_: _y, keep_prob: _keep_prob }

    train_images, test_images, train_labels, test_labels = prepare_data()
    etl8g_dataset = etl8g.Dataset(train_images, train_labels)

    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE])
        y_ = tf.placeholder(tf.float32, shape=[None, LABEL_NUM])
        x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        tf.summary.image('image', x_image, 100)

        global_step = tf.Variable(0, trainable=False)

        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, DECAY_STEPS, DECAY_RATE, staircase=True)
        keep_prob = tf.placeholder(tf.float32)

        logits = etl8g.inference(x_image, keep_prob)

        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('keep_prob', keep_prob)

        loss = etl8g.loss(logits, y_)
        tf.summary.scalar('loss', loss)

        train_step = etl8g.training(loss, FLAGS.optimizer, global_step, learning_rate)

        accuracy = etl8g.evaluation(logits, y_)
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)
            sess.run(init)
            batch_counter = 0

            start_time = time.time()

            print_stdout(("train_data_num %d, test_data_num %d batch_size %d max_epoch %d " + \
                         "start_learning_rate %0.5f optimizer %s train_keep_prob %0.5f " + \
                         "batch normalization %s") % \
                         (train_images.shape[0], test_images.shape[0], FLAGS.batch_num, FLAGS.max_epoch,
                          FLAGS.learning_rate, FLAGS.optimizer, FLAGS.train_keep_prob, FLAGS.bn))

            epoc_count = 0

            for epoch in range(FLAGS.max_epoch):
                total_batch = train_images.shape[0] // FLAGS.batch_num
                true_count = []
                costs = []

                for i in range(total_batch):
                    batch_x, batch_y = etl8g_dataset.next_batch(FLAGS.batch_num)
                    print(global_step)

                    sess.run(train_step, feed_dict=feed_dict(batch_x, batch_y, FLAGS.train_keep_prob))

                duration = time.time() - start_time

                perm = np.random.choice(train_images.shape[0], 100)
                validation_images = train_images[perm]
                validation_labels = train_labels[perm]

                train_accuracy = accuracy.eval(feed_dict=feed_dict(validation_images, validation_labels, 1.0))
                print_stdout("epoch %d, num_example %d, training accuracy %.03f (%.03f sec)" % \
                             (epoch, validation_images.shape[0], train_accuracy, duration))

                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(validation_images, validation_labels, 1.0))
                train_writer.add_summary(summary, epoch)
                perm2 = np.random.choice(test_images.shape[0], 100)
                sample_test_images = test_images[perm2]
                sample_test_labels = test_labels[perm2]

                test_feed_dict = feed_dict(sample_test_images, sample_test_labels, 1.0)
                test_accuracy = accuracy.eval(feed_dict=test_feed_dict)
                test_summary = sess.run(merged, feed_dict=test_feed_dict)

                test_writer.add_summary(test_summary, epoch)
                print_stdout("num_example %d, test accuracy %.03f" % (sample_test_images.shape[0], test_accuracy))

            duration = time.time() - start_time

            test_feed_dict = feed_dict(test_images[perm], test_labels, 1.0)
            test_accuracy = accuracy.eval(feed_dict=test_feed_dict)

            print_stdout("num_example %d, test accuracy %.03f (%.03f sec)" % (test_images.shape[0], accuracy.eval(
                         feed_dict(test_images, test_labels, 1.0)), duration))


def main(_):
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=40,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_num', type=int, default=16,
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
    parser.add_argument('--bn', type=str, default='true',
                        help='Batch Normalization')
    parser.add_argument('--data_argument', type=str, default='false',
                        help='Data Argumentation')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
