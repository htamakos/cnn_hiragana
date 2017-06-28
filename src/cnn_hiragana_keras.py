import numpy as np
import scipy.misc
import scipy.ndimage
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib.keras.python.keras.datasets import mnist
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.layers import Dropout, Activation, Flatten
from tensorflow.contrib.keras.python.keras.optimizers import Adagrad
from tensorflow.contrib.keras.python.keras.utils import np_utils
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import initializers
from tensorflow.contrib.keras.python.keras import callbacks
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator

from maxout import MaxoutDense

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

NPZ = 'np_hiragana_32.npz'
IMAGE_SIZE = 32
LABEL_NUM = 75

def load_data(test_size=0.2):
    dataset = np.load(NPZ)
    images = dataset['image']
    labels = dataset['label']

    X = np.zeros([images.shape[0], IMAGE_SIZE, IMAGE_SIZE])
    Y = np.zeros([labels.shape[0], LABEL_NUM])

    for i in range(images.shape[0]):
        img = scipy.misc.imresize(images[i], (IMAGE_SIZE, IMAGE_SIZE), mode='F')
        label = labels[i]
        X[i] = img
        Y[i] = label

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=test_size)

    X_train = X_train.reshape(X_train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
    X_test = X_test.reshape(X_test.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)

    return X_train, y_train, X_test, y_test

def kernel_initializer():
    #return initializers.random_normal(stddev=0.01)
    return 'he_normal'

def simple_model():
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(LABEL_NUM))
    model.add(Activation('softmax'))

    return model

def mk_model():
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                            kernel_initializer=kernel_initializer()))
    model.add(LeakyReLU(alpha=.1))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                            kernel_initializer=kernel_initializer()))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), kernel_initializer=kernel_initializer()))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), kernel_initializer=kernel_initializer()))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(MaxoutDense(output_dim=256, nb_feature=4))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(LABEL_NUM, kernel_initializer='he_uniform'))
    model.add(Activation('softmax'))

    return model

def mk_model_with_bn():
    model = Sequential()
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                            kernel_initializer=kernel_initializer()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                            kernel_initializer=kernel_initializer()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(filters=128, kernel_size=(3, 3),
                            kernel_initializer=kernel_initializer()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(filters=128, kernel_size=(3, 3),
                            kernel_initializer=kernel_initializer()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(filters=256, kernel_size=(3, 3),
                            kernel_initializer=kernel_initializer()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(filters=256, kernel_size=(3, 3),
                            kernel_initializer=kernel_initializer()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(LABEL_NUM, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))

    return model

if __name__ == '__main__':
    batch_size = 16
    nb_epoch = 400

    optimizer = 'adam'
    log_dir = './keras_logs/simple11'
    old_session = K.get_session()

    X_train, y_train, X_test, y_test = load_data()
    datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
    datagen.fit(X_train)

    with tf.Graph().as_default():
        session = tf.Session('')
        K.set_session(session)
        K.set_learning_phase(1)

        es_cb = callbacks.EarlyStopping(monitor='val_loss', patience=200, verbose=0, mode='auto')
        lr_cb = callbacks.LearningRateScheduler(lambda epoch: float(learning_rates[epoch]))

        model = mk_model_with_bn()
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        #model.fit(X_train, y_train,
        #      batch_size=batch_size, epochs=nb_epoch,
        #      verbose=1,
        #      validation_data=(X_test, y_test),
        #      callbacks=[tb_cb])
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                            epochs=nb_epoch, verbose=1, steps_per_epoch=X_train.shape[0],
                            validation_data=(X_test, y_test), callbacks=[tb_cb, es_cb, lr_cb])

        score = model.evaluate(X_test, y_test, verbose=0)
        print('\nTest score   : {:>.4f}'.format(score[0]))
        print('Test accuracy: {:>.4f}'.format(score[1]))

    K.clear_session()
