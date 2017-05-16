#!/bin/bash

nohup python cnn_hiragana.py \
--batch_num 16 \
--data_argument true \
--optimizer AdadeltaOptimizer \
--train_keep_prob 0.8 \
--log_dir ./cnn_hiragana_logs/exp10 \
> cnn_result_`date '+%F_%H_%M'`.log 2>&1 &
