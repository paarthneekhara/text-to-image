# downloads/extracts datasets described in the README.md

import os
import sys
import errno
import tarfile

if sys.version_info >= (3,):
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

DATA_DIR = 'Data'


# http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def create_data_paths():
    if not os.path.isdir(DATA_DIR):
        raise EnvironmentError('Needs to be run from project directory containing ' + DATA_DIR)
    needed_paths = [
        os.path.join(DATA_DIR, 'samples'),
        os.path.join(DATA_DIR, 'val_samples'),
        os.path.join(DATA_DIR, 'Models'),
    ]
    for p in needed_paths:
        make_sure_path_exists(p)


# adapted from http://stackoverflow.com/questions/51212/how-to-write-a-download-progress-indicator-in-python
def dl_progress_hook(count, blockSize, totalSize):
    percent = int(count * blockSize * 100 / totalSize)
    sys.stdout.write("\r" + "...%d%%" % percent)
    sys.stdout.flush()


def download_dataset(data_name):
    if data_name == 'flowers':
        print('== Flowers dataset ==')
        flowers_dir = os.path.join(DATA_DIR, 'flowers')
        flowers_jpg_tgz = os.path.join(flowers_dir, '102flowers.tgz')
        make_sure_path_exists(flowers_dir)

        # the original google drive link at https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view
        # from https://github.com/reedscot/icml2016 is problematic to download automatically, so included
        # the text_c10 directory from that archive as a bzipped file in the repo
        captions_tbz = os.path.join(DATA_DIR, 'flowers_text_c10.tar.bz2')
        print('Extracting ' + captions_tbz)
        captions_tar = tarfile.open(captions_tbz, 'r:bz2')
        captions_tar.extractall(flowers_dir)

        flowers_url = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
        print('Downloading ' + flowers_jpg_tgz + ' from ' + flowers_url)
        urlretrieve(flowers_url, flowers_jpg_tgz,
                    reporthook=dl_progress_hook)
        print('Extracting ' + flowers_jpg_tgz)
        flowers_jpg_tar = tarfile.open(flowers_jpg_tgz, 'r:gz')
        flowers_jpg_tar.extractall(flowers_dir)  # archive contains jpg/ folder

    elif data_name == 'skipthoughts':
        print('== Skipthoughts models ==')
        SKIPTHOUGHTS_DIR = os.path.join(DATA_DIR, 'skipthoughts')
        SKIPTHOUGHTS_BASE_URL = 'http://www.cs.toronto.edu/~rkiros/models/'
        make_sure_path_exists(SKIPTHOUGHTS_DIR)

        # following https://github.com/ryankiros/skip-thoughts#getting-started
        skipthoughts_files = [
            'dictionary.txt', 'utable.npy', 'btable.npy', 'uni_skip.npz', 'uni_skip.npz.pkl', 'bi_skip.npz',
            'bi_skip.npz.pkl',
        ]
        for filename in skipthoughts_files:
            src_url = SKIPTHOUGHTS_BASE_URL + filename
            print('Downloading ' + src_url)
            urlretrieve(src_url, os.path.join(SKIPTHOUGHTS_DIR, filename),
                        reporthook=dl_progress_hook)

    elif data_name == 'nltk_punkt':
        import nltk
        print('== NLTK pre-trained Punkt tokenizer for English ==')
        nltk.download('punkt')

    elif data_name == 'pretrained_model':
        print('== Pretrained model ==')
        MODEL_DIR = os.path.join(DATA_DIR, 'Models')
        pretrained_model_filename = 'latest_model_flowers_temp.ckpt'
        src_url = 'https://bitbucket.org/paarth_neekhara/texttomimagemodel/raw/74a4bbaeee26fe31e148a54c4f495694680e2c31/' + pretrained_model_filename
        print('Downloading ' + src_url)
        urlretrieve(
            src_url,
            os.path.join(MODEL_DIR, pretrained_model_filename),
            reporthook=dl_progress_hook,
        )

    else:
        raise ValueError('Unknown dataset name: ' + data_name)


def main():
    create_data_paths()
    # TODO: make configurable via command-line
    download_dataset('flowers')
    download_dataset('skipthoughts')
    download_dataset('nltk_punkt')
    download_dataset('pretrained_model')
    print('Done')


if __name__ == '__main__':
    main()
