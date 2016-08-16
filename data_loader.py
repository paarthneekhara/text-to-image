import json
from os.path import join, isfile
import re
import numpy as np
import pickle

def load_captions_data(data_dir):
	ic_train_file = join(data_dir, 'annotations/captions_train2014.json')
	ic_val_file = join(data_dir, 'annotations/captions_val2014.json')
	all_data_file = join(data_dir, 'ic_all_data.pkl')
	caption_dic_file = join(data_dir, 'ic_caption_dic.pkl')

	if isfile(all_data_file):
		with open(all_data_file) as f:
			all_data = pickle.load(f)
			print "Max length", all_data['max_caption_length']
			print "Total Words", len(all_data['caption_dic'])
			print "Total Captions", len(all_data['training_ic_data'])
			return all_data

	with open(ic_train_file) as f:
		ic_train = json.loads(f.read())
	with open(ic_val_file) as f:
		ic_val = json.loads(f.read())

	caption_dic, max_length = make_caption_word_dictionary(ic_train, ic_val)
	training_ic_data = extract_data(ic_train['annotations'], caption_dic, max_length)
	val_ic_data = extract_data(ic_val['annotations'], caption_dic, max_length)

	caption_dic_data = {
		'caption_dic' : caption_dic,
		'max_caption_length' : max_length
	}

	all_data = {
		'caption_dic' : caption_dic,
		'max_caption_length' : max_length,
		'training_ic_data' : training_ic_data,
		'val_ic_data' : val_ic_data
	}

	with open(all_data_file,'wb') as f:
		pickle.dump(all_data, f)
	with open(caption_dic_file, 'wb') as f:
		pickle.dump(caption_dic_data, f)

	print "Total Words", len(caption_dic)
	print "Training Data", len(training_ic_data)
	print "Validation Data", len(val_ic_data)
	return all_data

def make_caption_word_dictionary(ic_train, ic_val):
	word_regex = re.compile(r'\w+')
	caption_dic = {}
	max_length = 0
	for annotation in ic_train['annotations']:
		caption_words = re.findall(word_regex, annotation['caption'])
		if len(caption_words) > max_length: max_length = len(caption_words)
		for word in caption_words:
			caption_dic[word] = True

	for annotation in ic_val['annotations']:
		caption_words = re.findall(word_regex, annotation['caption'])
		if len(caption_words) > max_length: max_length = len(caption_words)
		for word in caption_words:
			caption_dic[word] = True

	caption_dic['UNK'] = True
	caption_dic_indexed = {}

	idx = 1
	for cap in caption_dic:
		caption_dic_indexed[cap] = idx
		idx += 1


	return caption_dic_indexed, max_length

def extract_data(ic_annotations, caption_dic, max_length):
	data = []
	word_regex = re.compile(r'\w+')
	for annotation in ic_annotations:
		caption = np.zeros(max_length)
		caption_words = re.findall(word_regex, annotation['caption'])

		idx = 0
		for i in range(max_length - len(caption_words), max_length):
			caption[i] = caption_dic[ caption_words[idx] ]
			idx += 1

		data.append({
			'image_id' : annotation['image_id'],
			'caption' : caption
		})
	return data

if __name__ == '__main__':
	load_captions_data('Data')

