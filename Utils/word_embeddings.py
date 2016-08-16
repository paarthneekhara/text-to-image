from os.path import join, isfile
import numpy as np
import pickle

def load_word_embeddings(data_dir, embedding_path):
	embedding_file = join(data_dir, embedding_path)
	embedding_word_vocab_file = join(data_dir, 'embedding_word_vocab.pkl')
	embedding_wts_file = join(data_dir, 'embedding_wts.npy')

	with open(embedding_file) as emb:
		embedding_data = [line.strip().split() for line in emb]
	
	word_vocab = {}
	n = len(embedding_data)
	emb_size = len(embedding_data[0]) - 1

	# +1 FOR PADDING 0s
	embedding_w = np.zeros((n+1,emb_size), dtype = 'float32')
	for i, row in enumerate(embedding_data):
		embedding_w[i,:] = np.asarray( row[1:], 'float32' )
		word_vocab[row[0]] = i

	print embedding_w.shape
	print len(word_vocab)

	return embedding_w, word_vocab
	

