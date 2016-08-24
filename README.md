# Text To Image Synthesis Using Thought Vectors

This is an experimental tensorflow implementation of synthesizing Images from captions using [Skip Thought Vectors][1]. The images are synthesized using the GAN-CLS Algorithm from the paper [Generative Adversarial Text-to-Image Synthesis][2]. This implementation is built on top of the excellent [DCGAN in Tensorflow][3]. The following is the model architecture. The blue bars represent the text encoding using Skip Thought Vectors.

![Model architecture](http://i.imgur.com/dNl2HkZ.jpg)

Image Source : [Generative Adversarial Text-to-Image Synthesis][2] Paper

## Requirements
- Python 2.7.6
- [Tensorflow][4]
- [h5py][5]
- [Theano][6] : for skip thought vectors
- [scikit-learn][7] : for skip thought vectors
- [NLTK][8] : for skip thought vectors

## Datasets
- The model is currently trained on the [flowers dataset][9]. Download the images from [this link][9] and save them in ```Data/flowers/jpg```. Also download the captions from [this link][10]. Extract the archive and copy the ```text_c_10``` folder and paste it in ```Data/flowers```.
- Download the pretrained models and vocabulary for skip thought vectors as per the instructions give here. Save the downloaded files in ```Data/skipthoughts```.
- Make empty directories in Data, ```Data/samples```,  ```Data/val_samples``` and ```Data/Models```. They will be used for sampling the generated images, while training.

## Usage
- <b>Data Processing</b> : Extract the skip thought vectors for the flowers data set using :
```
python data_loader.py --data_set="flowers"
```
- <b>Training</b>
  * Basic usage `python train.py --data_set="flowers"`
  * Options
      - `z_dim`: Noise Dimension. Default is 100.
      - `t_dim`: Text feature dimension. Default is 256.
      - `batch_size`: Batch Size. Default is 64.
      - `image_size`: Image dimension. Default is 64.
      - `gf_dim`: Number of conv in the first layer generator. Default is 64.
      - `df_dim`: Number of conv in the first layer discriminator. Default is 64.
      - `gfc_dim`: Dimension of gen untis for for fully connected layer. Default is 1024.
      - `caption_vector_length`: Length of the caption vector. Default is 1024.
      - `data_dir`: Data Directory. Default is `Data/`.
      - `learning_rate`: Learning Rate. Default is 0.0002.
      - `beta1`: Momentum for adam update. Default is 0.5.
      - `epochs`: Max number of epochs. Default is 600.
      - `resume_model`: Resume training from a pretrained model path.
      - `data_set`: Data Set to train on. Default is flowers.
      
- <b>Generating Images from Captions</b>
  * Write the captions in text file, and save it as ```Data/sample_captions.txt```. Generate the skip thought vectors for these captions using:
  ```
  python generate_thought_vectors.py --caption_file="Data/sample_captions.txt"
  ```
  * Generate the Images for the thought vectors using:
  ```
  python generate_images.py --model_path=<path to the trained model>
  ```


[1]:http://arxiv.org/abs/1506.06726
[2]:http://arxiv.org/abs/1605.05396
[3]:https://github.com/carpedm20/DCGAN-tensorflow
[4]:https://github.com/tensorflow/tensorflow
[5]:http://www.h5py.org/
[6]:https://github.com/Theano/Theano
[7]:http://scikit-learn.org/stable/index.html
[8]:http://www.nltk.org/
[9]:http://www.robots.ox.ac.uk/~vgg/data/flowers/102/
[10]:https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view
