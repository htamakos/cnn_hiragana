import sys
import os
import struct
import argparse

from PIL import Image
from PIL import ImageOps
import numpy as np

FLAGS = None

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

# 1ファイルに格納されているデータ数
NUM_DATASET = 4780

HIRAGANA_DATA_DIR = DATA_DIR + '/hiragana_images/'
NPZ = '../data/ETL8G/np_hiragana.npz'
LABELS = os.listdir(HIRAGANA_DATA_DIR)
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
    # 16階調のグレースケールを255階調のグレースケールへ変換する
    image = Image.eval(iL, lambda x: int(255 * x /16))
    return r + (image, )

def create_hiragana_image():
    """
    Args:
      None
    Returns:
      np.array
    """
    dirname = DATA_DIR + '/hiragana_images'

    for i in range(1, 32):
        filename = "{}/ETL8G_{:02d}".format(DATA_DIR, i)

        with open(filename, 'rb') as f:
            for j in range(1, NUM_DATASET):
                r = read_record_ETL8G(f)
                hiragana_jiscode = hex(r[1])

                # ひらがなのみ（0x24**）のみ処理をする
                if '0x24' in hiragana_jiscode:
                    image = r[-1]
                    jiscode_dirname = dirname + '/' + hiragana_jiscode

                    if not hiragana_jiscode in os.listdir(dirname):
                        os.mkdir(jiscode_dirname)
                    image_filename = "{}/ETL8G_{:02d}_{}.png".format(jiscode_dirname, i, j)
                    image.save(image_filename)
                    print("create " + image_filename)

def _find_all_files(path):
    dirnames = os.listdir(path)
    files = []
    for dirname in dirnames:
        filenames = os.listdir(path + "/" + dirname)
        files += filenames
    return files


def create_npz(data_argumentation=True):
    """
    numpyの圧縮ファイル形式で保存する関数
    """
    images = []
    labels = []
    for i, l in enumerate(LABELS):
        directory = HIRAGANA_DATA_DIR + l + '/'
        files = os.listdir(directory)
        label = np.zeros([NUM_LABELS])
        label_index = LABELS.index(l)
        label[label_index] = 1

        for file in files:
            try:
                img = Image.open(directory + file)
            except:
                print("Skip a corruputed file: ", file)
                continue
            img = img.resize(IMAGE_PIXCELS)

            # 元の画像のピクセル・ラベル追加
            pixels = np.array(img.getdata())
            images.append(pixels/255.0)
            labels.append(label)

            if data_argumentation:
                # 元の画像を+15度回転
                rotate_img = img.rotate(15)
                pixels = np.array(rotate_img.getdata())
                images.append(pixels/255.0)
                labels.append(label)

                # 元の画像を-15度回転
                m_rotate_img = img.rotate(-15)
                pixels = np.array(m_rotate_img.getdata())
                images.append(pixels/255.0)
                labels.append(label)

                # 文字色をはっきりと
                img2 = img.point(lambda x: x * 5.0)
                pixels = np.array(img2.getdata())
                images.append(pixels/255.0)
                labels.append(label)

                # 文字色を薄くする
                img3 = img.point(lambda x: x * 0.5)
                pixels = np.array(img3.getdata())
                images.append(pixels/255.0)
                labels.append(label)

    np.savez(DATA_DIR + "/np_hiragana.npz", image=images, label=labels)

def setup_dir():
    """
    DATA_DIR/hiragana_imagesディレクトリが作成されていない場合に作成する関数
    """

    if not 'hiragana_images' in os.listdir(DATA_DIR):
        dirname = DATA_DIR + '/hiragana_images'
        os.mkdir(dirname)
        print('Create ' + dirname)

def main():
    if FLAGS.process == 'create_hiragana_image':
        setup_dir()
        create_hiragana_image()
    elif FLAGS.process == 'create_npz':
        setup_dir()
        create_npz()
    else:
        print('create_hiragana_image または create_npz を入力する必要があります。')
        sys.exit(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--process',
        type=str,
        default=None,
        help='実行する処理を選択します。[create_hiragana_image, create_npz]'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
