import json
from os.path import join, isfile
import re
import numpy as np
import pickle
from Utils import word_embeddings

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
			print "Total Captions", len(all_data['training_ic_data'])
			return all_data

	with open(ic_train_file) as f:
		ic_train = json.loads(f.read())
	with open(ic_val_file) as f:
		ic_val = json.loads(f.read())

	embedding_w, embeddingg_vocab = word_embeddings.load_word_embeddings(data_dir, 'glove.6B/glove.6B.50d.txt')
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


if __name__ == '__main__':
	load_captions_data('Data')

