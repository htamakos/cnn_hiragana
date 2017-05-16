import sys
import os
import struct
import argparse

from PIL import Image
from PIL import ImageOps
import numpy as np

# １つの画像データのサイズ（バイト）
RECORD_SIZE = 8199
# struct.unpackで指定するフォーマット文字列
# struct.unpack(fmt, buffer)
#   バッファ buffer を、書式文字列 fmt に従ってアンパックします
# >  : バイトオーダがビックエンディアン、サイズがstandard、アライメントがnoneであることを指定する
# H  : unsigned short
# s  : char[]
# I  : unsigned int
# B  : unsigned char
# x  : パディングバイト
# 詳細は http://etlcdb.db.aist.go.jp/?page_id=2461 を参照
RECORD_FMT = '>2H8sI4B4H2B30x8128s11x'

# 1つの画像の縦横のサイズ(x, y)
IMAGE_SIZE_X = 128
IMAGE_SIZE_Y = 127
IMAGE_SIZES = (IMAGE_SIZE_X, IMAGE_SIZE_Y)

# Fは32-bit floating point pixels
# http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#concept-modes
IMAGE_MODE = 'F'
# unpackした後の戻り値から画像データのみを抽出するために使用するインデックス
IMAGE_INDEX = 14

# http://pillow.readthedocs.io/en/3.4.x/handbook/writing-your-own-file-decoder.html#file-decoders
DECODER_NAME = 'bit'
# 1つの画像のグレースケールの階調(ビット数)"
## つまり 2**4 = 16階調
BITS_NUM = 4
# ETLの画像データが格納されているディレクトリ
DATA_DIR = '../data/ETL8G'
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

# 1ファイルに格納されているデータ数
NUM_DATASET = 4780

HIRAGANA_DATA_DIR = DATA_DIR + '/hiragana_images/'
NPZ = '../data/ETL8G/np_hiragana.npz'
NUM_LABELS = len(LABELS)
IMAGE_SIZE = 28
IMAGE_PIXCELS = [IMAGE_SIZE, IMAGE_SIZE]

def read_record_ETL8G(f):
    """ ETL8G
    Args:
      f: file object
    Returns:
      tubple: ex. (1, 9250, b'A.HIRA  ', 1, 0, 0, 1, 24, 3552, 0, 8001, 16880, 0, 0, b'...',
                   <PIL.Image.Image image mode=L size=128x127 at 0x7F134FAE9198>)
    Raises:
      None
    """
    # ファイルオブジェクトからRECORD_SIZEバイト分読み出す
    s = f.read(RECORD_SIZE)
    # RECORD_FMTというバイナリフォーマットに従ってシリアライズする
    r = struct.unpack(RECORD_FMT, s)
    # 32ビット浮動小数点形式のImageオブジェクトに変換
    iF = Image.frombytes(IMAGE_MODE, IMAGE_SIZES, r[IMAGE_INDEX], DECODER_NAME, BITS_NUM)
    # グレースケールの画像に変換
    iL = iF.convert('L')
    return r + (iL, )

def read_hiragana():
    images = []
    labels = []

    dirname = DATA_DIR + '/hiragana_images'

    for i in range(1, 32):
        filename = "{}/ETL8G_{:02d}".format(DATA_DIR, i)

        with open(filename, 'rb') as f:
            for j in range(1, NUM_DATASET):
                r = read_record_ETL8G(f)
                hiragana_jiscode = hex(r[1])

                # ひらがなのみ（0x24**）のみ処理をする
                if '0x24' in hiragana_jiscode:
                    images.append(np.array(r[-1]))
                    label = np.zeros([len(LABELS)], np.uint8)
                    label_index = LABELS.index(hiragana_jiscode)
                    label[label_index] = 1
                    labels.append(label)

    np.savez('hiragana.npz', image=images, label=labels)

read_hiragana()
