# ひらがな画像認識

[ETL8Gデータ・セット](http://etlcdb.db.aist.go.jp/?page_id=2461) を用いて
CNNによる画像認識を行なうPythonスクリプトです。

## environment

```
$ cat /etc/oracle-release
Oracle Linux Server release 7.3

$ lscpu
アーキテクチャ: x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                4
On-line CPU(s) list:   0-3
コアあたりのスレッド数:2
ソケットあたりのコア数:2
Socket(s):             1
NUMAノード:         1
ベンダーID:        GenuineIntel
CPUファミリー:    6
モデル:             58
Model name:            Intel(R) Core(TM) i5-3320M CPU @ 2.60GHz
ステッピング:    9
CPU MHz:               3099.992
BogoMIPS:              5188.19
仮想化:             VT-x
L1d キャッシュ:   32K
L1i キャッシュ:   32K
L2 キャッシュ:    256K
L3 キャッシュ:    3072K
NUMAノード 0 CPU:   0-3

$ cat /proc/meminfo
MemTotal:        7868556 kB
MemFree:          164040 kB
MemAvailable:    5126012 kB
Buffers:               0 kB
Cached:          4717272 kB
SwapCached:       169376 kB
Active:          3206044 kB
Inactive:        3380788 kB
Active(anon):     833216 kB
Inactive(anon):  1166996 kB
Active(file):    2372828 kB
Inactive(file):  2213792 kB
Unevictable:          32 kB
Mlocked:              32 kB
SwapTotal:       7996412 kB
SwapFree:        7364908 kB
Dirty:                 4 kB
Writeback:             0 kB
AnonPages:       1720528 kB
Mapped:           132468 kB
Shmem:            130652 kB
Slab:             823216 kB
SReclaimable:     628768 kB
SUnreclaim:       194448 kB
KernelStack:       26320 kB
PageTables:        53652 kB
NFS_Unstable:          0 kB
Bounce:                0 kB
WritebackTmp:          0 kB
CommitLimit:    11930688 kB
Committed_AS:   14349616 kB
VmallocTotal:   34359738367 kB
VmallocUsed:      361380 kB
VmallocChunk:   34359289504 kB
HardwareCorrupted:     0 kB
AnonHugePages:    993280 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
DirectMap4k:      379072 kB
DirectMap2M:     7698432 kB
```

## Instaration

```
$ pip install numpy
$ pip install scipy
$ pip install sklearn
$ pip install tensorflow
```

## Usage

```
$ cd src
$ python input_data.py
$ sh build.sh
```

## Result

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 30, 30, 32)        320
_________________________________________________________________
activation_1 (Activation)    (None, 30, 30, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 32)        9248
_________________________________________________________________
activation_2 (Activation)    (None, 28, 28, 32)        0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 32)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 14, 32)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 12, 64)        18496
_________________________________________________________________
activation_3 (Activation)    (None, 12, 12, 64)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 10, 10, 64)        36928
_________________________________________________________________
activation_4 (Activation)    (None, 10, 10, 64)        0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0
_________________________________________________________________
dropout_2 (Dropout)          (None, 5, 5, 64)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 1600)              0
_________________________________________________________________
dense_1 (Dense)              (None, 256)               409856
_________________________________________________________________
activation_5 (Activation)    (None, 256)               0
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0
_________________________________________________________________
dense_2 (Dense)              (None, 75)                19275
_________________________________________________________________
activation_6 (Activation)    (None, 75)                0
=================================================================
Total params: 494,123.0
Trainable params: 494,123.0
Non-trainable params: 0.0
_________________________________________________________________
Train on 9275 samples, validate on 2319 samples
Epoch 1/20
9275/9275 [==============================] - 43s - loss: 3.9201 - acc: 0.0774 - val_loss: 2.5601 - val_acc: 0.4588
Epoch 2/20
9275/9275 [==============================] - 42s - loss: 2.5701 - acc: 0.3298 - val_loss: 1.3239 - val_acc: 0.6809
Epoch 3/20
9275/9275 [==============================] - 42s - loss: 1.8105 - acc: 0.4967 - val_loss: 0.8331 - val_acc: 0.7715
Epoch 4/20
9275/9275 [==============================] - 42s - loss: 1.3546 - acc: 0.6000 - val_loss: 0.6609 - val_acc: 0.8284
Epoch 5/20
9275/9275 [==============================] - 42s - loss: 1.0929 - acc: 0.6746 - val_loss: 0.5079 - val_acc: 0.8525
Epoch 6/20
9275/9275 [==============================] - 42s - loss: 0.8884 - acc: 0.7344 - val_loss: 0.3885 - val_acc: 0.8892
Epoch 7/20
9275/9275 [==============================] - 42s - loss: 0.7499 - acc: 0.7653 - val_loss: 0.3345 - val_acc: 0.8974
Epoch 8/20
9275/9275 [==============================] - 42s - loss: 0.6575 - acc: 0.7944 - val_loss: 0.2927 - val_acc: 0.9017
Epoch 9/20
9275/9275 [==============================] - 42s - loss: 0.5835 - acc: 0.8102 - val_loss: 0.2471 - val_acc: 0.9207
Epoch 10/20
9275/9275 [==============================] - 42s - loss: 0.5269 - acc: 0.8339 - val_loss: 0.2137 - val_acc: 0.9276
Epoch 11/20
9275/9275 [==============================] - 42s - loss: 0.4665 - acc: 0.8534 - val_loss: 0.1960 - val_acc: 0.9383
Epoch 12/20
9275/9275 [==============================] - 42s - loss: 0.4078 - acc: 0.8685 - val_loss: 0.2010 - val_acc: 0.9267
Epoch 13/20
9275/9275 [==============================] - 42s - loss: 0.3945 - acc: 0.8798 - val_loss: 0.1850 - val_acc: 0.9375
Epoch 14/20
9275/9275 [==============================] - 42s - loss: 0.3645 - acc: 0.8839 - val_loss: 0.1768 - val_acc: 0.9426
Epoch 15/20
9275/9275 [==============================] - 42s - loss: 0.3284 - acc: 0.8954 - val_loss: 0.1683 - val_acc: 0.9366
Epoch 16/20
9275/9275 [==============================] - 42s - loss: 0.3115 - acc: 0.8950 - val_loss: 0.1558 - val_acc: 0.9422
Epoch 17/20
9275/9275 [==============================] - 42s - loss: 0.2996 - acc: 0.9030 - val_loss: 0.1519 - val_acc: 0.9474
Epoch 18/20
9275/9275 [==============================] - 42s - loss: 0.2783 - acc: 0.9108 - val_loss: 0.1402 - val_acc: 0.9457
Epoch 19/20
9275/9275 [==============================] - 42s - loss: 0.2565 - acc: 0.9142 - val_loss: 0.1318 - val_acc: 0.9513
Epoch 20/20
9275/9275 [==============================] - 42s - loss: 0.2427 - acc: 0.9247 - val_loss: 0.1314 - val_acc: 0.9517

Test score   : 0.1314
Test accuracy: 0.9517
```
