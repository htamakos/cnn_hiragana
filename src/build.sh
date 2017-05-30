#!/bin/bash

nohup python cnn_hiragana_keras.py > cnn_hiragana_keras_result_`date '+%F_%H_%M'`.log 2>&1 &
