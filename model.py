import tensorflow as tf
from Utils import ops

class GAN:
	'''
	OPTIONs
	Wemb : Pretrained Word Embeddings
	z_dim : Noise dimension
	t_dim : Text feature dimension
	image_size : Image Dimension 64
	gf_dim : Number of conv in the first layer generator 64
	df_dim : Number of conv in the first layer discriminator 64
	gfc_dim : Dimension of gen untis for for fully connected layer 1024
	'''
	def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
		return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

	def init_bias(self, dim_out, name=None):
		return tf.Variable(tf.zeros([dim_out]), name=name)

	def __init__(self, options):
		self.options = options
		self.Wemb = tf.constant(options['Wemb'], dtype='float32', name='Wemb')

		self.lstm_W = []
		self.lstm_U = []
		self.lstm_b = []
		for i in range(options['num_lstm_layers']):
			W = self.init_weight(options['rnn_size'], 4 * options['rnn_size'], name = ('rnnw_' + str(i)))
			U = self.init_weight(options['rnn_size'], 4 * options['rnn_size'], name = ('rnnu_' + str(i)))
			b = self.init_bias(4 * options['rnn_size'], name = ('rnnb_' + str(i)))
			self.lstm_W.append(W)
			self.lstm_U.append(U)
			self.lstm_b.append(b)

		self.g_bn0 = ops.batch_norm(name='g_bn0')
		self.g_bn1 = ops.batch_norm(name='g_bn1')
		self.g_bn2 = ops.batch_norm(name='g_bn2')
		self.g_bn3 = ops.batch_norm(name='g_bn3')


	# REUSED GENERATOR IMPLEMENTATION : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def generator(self, t_z):
		s = self.options['image_size']
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

		self.z_, self.h0_w, self.h0_b = ops.linear(t_z, self.options['gf_dim']*8*s16*s16, 'g_h0_lin', with_w=True)
		self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = tf.nn.relu(self.g_bn0(self.h0))

		h1, self.h1_w, self.h1_b = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4],
												 name='g_h1', with_w=True)
		h1 = tf.nn.relu(self.g_bn1(h1))

		h2, self.h2_w, self.h2_b = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2],
												name='g_h2', with_w=True)
		h2 = tf.nn.relu(self.g_bn2(h2))

		h3, self.h3_w, self.h3_b = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1],
												name='g_h3', with_w=True)
		h3 = tf.nn.relu(self.g_bn3(h3))

		h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')

		return tf.tanh(h4)

	# REUSED DISCRIMINATOR IMPLEMENTATION : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def discriminator(self, image):
		h0 = ops.lrelu(ops.conv2d(image, self.options['df_dim'], name = 'd_h0_conv'))
		h1 = ops.lrelu(ops.conv2d(h0, self.options['df_dim']*2, name = 'd_h1_conv'))
		h2 = ops.lrelu(ops.conv2d(h1, self.options['df_dim']*4, name = 'd_h2_conv'))
		h3 = ops.lrelu(ops.conv2d(h2, self.options['df_dim']*8, name = 'd_h3_conv'))
		h4 = ops.linear(tf.reshape(h3, [self.options['batch_size'], -1]), 1, 'd_h3_lin')

		return tf.nn.sigmoid(h4), h4