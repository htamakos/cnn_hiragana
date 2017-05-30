#!/bin/bash

nohup python cnn_hiragana.py \
--batch_num 50 \
--data_argument false \
--optimizer AdamOptimizer \
--train_keep_prob 0.8 \
--log_dir ./cnn_hiragana_logs/exp7 \
> cnn_result_`date '+%F_%H_%M'`.log 2>&1 &
