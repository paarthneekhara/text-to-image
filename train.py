import tensorflow as tf
import numpy as np
import model
import argparse
import pickle
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--z_dim', type=int, default=100,
					   help='z_dim')
	parser.add_argument('--t_dim', type=int, default=256,
					   help='t_dim')
	parser.add_argument('--batch_size', type=int, default=64,
					   help='batch_size')
	parser.add_argument('--image_size', type=int, default=64,
					   help='image_size')
	parser.add_argument('--gf_dim', type=int, default=64,
					   help='gf_dim')
	parser.add_argument('--df_dim', type=int, default=64,
					   help='df_dim')
	parser.add_argument('--gfc_dim', type=int, default=1024,
					   help='gfc_dim')
	parser.add_argument('--caption_vector_length', type=int, default=4800,
					   help='caption_vector_length')

	parser.add_argument('--data_dir', type=str, default="Data",
					   help='data_dir')

	parser.add_argument('--learning_rate', type=float, default=0.0002,
					   help='learning_rate')
	parser.add_argument('--beta1', type=float, default=0.5,
					   help='beta1')

	parser.add_argument('--epochs', type=int, default=100,
					   help='epochs')
	parser.add_argument('--resume_model', type=str, default=None,
                       help='Trained Model Path')


	args = parser.parse_args()
	model_options = {
		'z_dim' : args.z_dim,
		't_dim' : args.t_dim,
		'batch_size' : args.batch_size,
		'image_size' : args.image_size,
		'gf_dim' : args.gf_dim,
		'df_dim' : args.df_dim,
		'gfc_dim' : args.gfc_dim,
		'caption_vector_length' : args.caption_vector_length
	}
	
	with open(join(args.data_dir, 'meta_train.pkl')) as f:
		meta_data = pickle.load(f)

	gan = model.GAN(model_options)
	input_tensors, variables, loss, outputs = gan.build_model()

	d_optim = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
	g_optim = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])

	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	saver = tf.train.Saver()
	if args.resume_model:
		saver.restore(sess, args.resume_model)

	for i in range(args.epochs):
		batch_no = 0
		while batch_no*args.batch_size < meta_data['data_length']:
			real_images, wrong_images, caption_vectors, z_noise = get_training_batch(batch_no, args.batch_size, 
				args.image_size, args.z_dim, 'train', args.data_dir)
			_, _, d_loss, g_loss, gen = sess.run([d_optim, g_optim, loss['d_loss'], loss['g_loss'], outputs['generator']],
				feed_dict = {
					input_tensors['t_real_image'] : real_images,
					input_tensors['t_wrong_image'] : wrong_images,
					input_tensors['t_real_caption'] : caption_vectors,
					input_tensors['t_z'] : z_noise,
				})
			batch_no += 1
			if (batch_no % 100) == 0:
				print "Saving Images, Model"
				save_for_vis(args.data_dir, real_images, gen)
				save_path = saver.save(sess, "Data/Models/model_temp.ckpt")
			print d_loss, g_loss, batch_no, i
		save_path = saver.save(sess, "Data/Models/model_epoch{}.ckpt".format(i))

def save_for_vis(data_dir, real_images, generated_images):
	
	for i in range(0, real_images.shape[0]):
		real_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		real_images_255 = real_images[i,:,:,:] * 255
		scipy.misc.imsave( join(data_dir, 'samples/real_image_{}.jpg'.format(i)) , real_images_255)

		fake_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		fake_images_255 = generated_images[i,:,:,:] * 255
		scipy.misc.imsave(join(data_dir, 'samples/fake_image_{}.jpg'.format(i)), fake_images_255)


def get_training_batch(batch_no, batch_size, image_size, z_dim, split, data_dir):
	with h5py.File( join(data_dir, 'tvs/'+split + '_tvs_' + str(batch_no))) as hf:
		caption_vectors = np.array(hf.get('tv'))
	with h5py.File( join(data_dir, 'tvs/'+split + '_tv_image_id_' + str(batch_no))) as hf:
		image_ids = np.array(hf.get('tv'))

	real_images = np.zeros((batch_size, 64, 64, 3))
	wrong_images = np.zeros((batch_size, 64, 64, 3))
	
	for idx, image_id in enumerate(image_ids):
		image_file = join(data_dir, '%s2014/COCO_%s2014_%.12d.jpg'%(split, split, image_id) )
		image_array = image_processing.load_image_array(image_file, image_size)
		real_images[idx,:,:,:] = image_array

	for i in range(0, batch_size):
		wrong_images[i,:,:,:] = real_images[batch_size-i-1,:,:,:]

	z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])

	return real_images, wrong_images, caption_vectors, z_noise




if __name__ == '__main__':
	main()
