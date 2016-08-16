import numpy as np
from scipy import misc

def load_image_array(image_file, image_size):
	img = misc.imread(image_file)
	# GRAYSCALE
	if len(img.shape) == 2:
		img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'float32')
		img_new[:,:,0] = img
		img_new[:,:,1] = img
		img_new[:,:,2] = img
		img = img_new

	img_resized = misc.imresize(img, (image_size, image_size))
	return ((img_resized/255.0)-1).astype('float32')