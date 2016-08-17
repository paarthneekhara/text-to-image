import json
from os.path import join, isfile
import re
import numpy as np
import pickle
from Utils import word_embeddings
from Utils import image_processing
import argparse
import gc

def load_captions_data(data_dir):
	ic_train_file = join(data_dir, 'annotations/captions_train2014.json')
	ic_val_file = join(data_dir, 'annotations/captions_val2014.json')
	all_data_file = join(data_dir, 'ic_all_data.pkl')
	meta_data_file = join(data_dir, 'meta_data.pkl')

	if isfile(all_data_file):
		with open(all_data_file) as f:
			all_data = pickle.load(f)
			print "Max length", all_data['max_caption_length']
			print "Total Words", len(all_data['caption_dic']), all_data['word_embedding_lookup'].shape
			print "Total Training Captions", len(all_data['training_ic_data'])
			return all_data

	with open(ic_train_file) as f:
		ic_train = json.loads(f.read())
	with open(ic_val_file) as f:
		ic_val = json.loads(f.read())

	embedding_w, embeddingg_vocab = word_embeddings.load_word_embeddings(data_dir, 'glove.6B/glove.6B.300d.txt')
	caption_dic, word_embedding_lookup, max_length = make_caption_word_dictionary(ic_train, ic_val, embeddingg_vocab, embedding_w)
	print "Len caption dic", len(caption_dic)
	

	common_words = 0
	for key in caption_dic:
		if key.lower() in embeddingg_vocab:
			common_words += 1
	print "Coverage=", 1.0 * common_words/len(caption_dic)

	training_ic_data = extract_data(ic_train['annotations'], caption_dic, embeddingg_vocab, max_length)
	val_ic_data = extract_data(ic_val['annotations'], caption_dic, embeddingg_vocab, max_length)

	
	all_data = {
		'caption_dic' : caption_dic,
		'word_embedding_lookup' : word_embedding_lookup,
		'max_caption_length' : max_length,
		'training_ic_data' : training_ic_data,
		'val_ic_data' : val_ic_data
	}

	meta_data = {
		'max_caption_length' : max_length,
		'n_training' : len(training_ic_data),
		'n_val' : len(val_ic_data),
		'caption_dic' : caption_dic,
		'word_embedding_lookup' : word_embedding_lookup
	}


	with open(all_data_file,'wb') as f:
		pickle.dump(all_data, f)

	with open(meta_data_file,'wb') as f:
		pickle.dump(meta_data, f)
	
	print "Total Words", len(caption_dic)
	print "Training Data", len(training_ic_data)
	print "Validation Data", len(val_ic_data)
	return all_data
	# return all_data

def make_caption_word_dictionary(ic_train, ic_val, embeddingg_vocab, embedding_w):
	word_regex = re.compile(r'\w+')
	caption_dic = {}
	max_length = 0
	for annotation in ic_train['annotations']:
		caption_words = re.findall(word_regex, annotation['caption'])
		if len(caption_words) > max_length: max_length = len(caption_words)
		for word in caption_words:
			caption_dic[word.lower()] = True

	for annotation in ic_val['annotations']:
		caption_words = re.findall(word_regex, annotation['caption'])
		if len(caption_words) > max_length: max_length = len(caption_words)
		for word in caption_words:
			caption_dic[word.lower()] = True

	word_embedding_lookup = np.zeros( (len(caption_dic)+1, embedding_w.shape[1])) 
	
	idx = 0
	for word in caption_dic:
		caption_dic[word] = idx
		if word in embeddingg_vocab:
			word_embedding_lookup[idx,:] = embedding_w[ embeddingg_vocab[word] ]
		idx += 1

	print "Sanity Checck"
	
	print caption_dic['and']
	print word_embedding_lookup[caption_dic['and']]
	print embeddingg_vocab['and']
	print embedding_w[embeddingg_vocab['and']]

	return caption_dic, word_embedding_lookup, max_length

def extract_data(ic_annotations, caption_dic, embeddingg_vocab, max_length):
	import h5py
	data = []
	word_regex = re.compile(r'\w+')
	for annotation in ic_annotations:
		# FOR ZERO PADDING
		caption_array = [ len(caption_dic) for i in range(max_length)]
		caption = np.asarray(caption_array, dtype = 'int32')
		caption_words = re.findall(word_regex, annotation['caption'])

		idx = 0
		valid_caption = True
		for i in range(max_length - len(caption_words), max_length):
			caption_words[idx] = caption_words[idx].lower()
			if not caption_words[idx] in embeddingg_vocab:
				valid_caption = False
				break
			
			caption[i] = caption_dic[ caption_words[idx] ]
			idx += 1

		if valid_caption:
			data.append({
				'image_id' : annotation['image_id'],
				'caption' : caption
			})

	print "Extracted Data", len(data)
	print "Raw Data", len(ic_annotations)
	return data

# def get_training_batch(ic_data, batch_no, batch_size):
# 	start_index = batch_no * batch_size
# 	end_index = start_index + batch_size

# 	batch = {
# 		'caption' : 
# 	}
# 	for idx in range(start_index, end_index):
# 		new_idx = idx % len(ic_data)


def load_images(data_dir, split, image_size):
	import h5py
	captions_data = load_captions_data(data_dir)
	if split == 'train':
		captions = captions_data['training_ic_data']
	else:
		captions = captions_data['val_ic_data']
	
	image_ids = {}
	for cap in captions:
		image_ids[cap['image_id']] = True
	print len(image_ids)
	image_id_list = []
	gc.collect()
	image_features = np.zeros((len(image_ids), image_size, image_size, 3))
	idx = 0
	print "Shae",image_features.shape

	print "Total Images", len(image_ids)
	for image_id in image_ids:
		image_file = join(data_dir, '%s2014/COCO_%s2014_%.12d.jpg'%(split, split, image_id) )
		image_array = image_processing.load_image_array(image_file, image_size)
		image_features[idx,:,:,:] = image_array
		image_id_list.append(image_id)
		print "Progress..", 1.0 * idx/len(image_ids), idx
		idx += 1
		

	h5f_image_features = h5py.File( join(data_dir, split + '_image_features.h5'), 'w')
	h5f_image_features.create_dataset('image_features', data=image_features)
	h5f_image_features.close()

	with open( join(data_dir, split+'_image_id_list.pkl'), 'wb') as f:
		pickle.dump(image_id_list, f)



# FOR SANITY CHECK
# def reconstruct_caption(caption, caption_words, caption_dic, word_embedding_lookup):
# 	caption_dic_indexed = {}
# 	for word in caption_dic:
# 		caption_dic_indexed[caption_dic[word]] = word

# 	print caption
# 	print caption_words
# 	for c in caption:
# 		if c in caption_dic_indexed:
# 			print caption_dic_indexed[c], word_embedding_lookup[c]
# 		else:
# 			print word_embedding_lookup[c]

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, default='train',
                       help='train/val')
	parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
	parser.add_argument('--image_size', type=str, default=64,
                       help='Image Size')

	args = parser.parse_args()
	load_images(args.data_dir, args.split, args.image_size)

if __name__ == '__main__':
	main()

