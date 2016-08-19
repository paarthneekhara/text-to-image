import json
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import skipthoughts

def save_caption_vectors(data_dir, split, batch_size):
	import h5py

	meta_data = {}
	ic_file = join(data_dir, 'annotations/captions_{}2014.json'.format(split))
	with open(ic_file) as f:
		ic_data = json.loads(f.read())

	meta_data['data_length'] = len(ic_data['annotations'])
	with open(join(data_dir, 'meta_{}.pkl'.format(split)), 'wb') as f:
		pickle.dump(meta_data, f)

	model = skipthoughts.load_model()
	batch_no = 0
	print "Total Batches", len(ic_data['annotations'])/batch_size

	while batch_no*batch_size < len(ic_data['annotations']):
		captions = []
		image_ids = []
		idx = batch_no
		for i in range(batch_no*batch_size, (batch_no+1)*batch_size):
			idx = i%len(ic_data['annotations'])
			captions.append(ic_data['annotations'][idx]['caption'])
			image_ids.append(ic_data['annotations'][idx]['image_id'])

		print captions
		print image_ids
		# Thought Vectors
		tv_batch = skipthoughts.encode(model, captions)
		h5f_tv_batch = h5py.File( join(data_dir, 'tvs/'+split + '_tvs_' + str(batch_no)), 'w')
		h5f_tv_batch.create_dataset('tv', data=tv_batch)
		h5f_tv_batch.close()

		h5f_tv_batch_image_ids = h5py.File( join(data_dir, 'tvs/'+split + '_tv_image_id_' + str(batch_no)), 'w')
		h5f_tv_batch_image_ids.create_dataset('tv', data=image_ids)
		h5f_tv_batch_image_ids.close()

		print "Batches Done", batch_no, len(ic_data['annotations'])/batch_size
		batch_no += 1

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, default='train',
                       help='train/val')
	parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
	parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch Size')
	args = parser.parse_args()
	
	save_caption_vectors(args.data_dir, args.split, args.batch_size)

if __name__ == '__main__':
	main()