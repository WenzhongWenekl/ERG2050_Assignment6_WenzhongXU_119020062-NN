# -*- coding: utf-8 -*-
'''
Created on 2018��9��13��

@author: WQ
'''
from seqlib import *
from data_process import *

# reload(sys)
# sys.setdefaultencoding('utf-8')

def normalizecorpus():
    input = codecs.open('train.txt','r')
    output = codecs.open('corpus.txt','w')
    for line in input.readlines():
        output.write(' '.join(remove_list_tag(line.strip().split())))
        output.write("\n")
    input.close()
    output.close()

# normalizecorpus()
corpuspath = 'train.txt'
standardcorpus = 'corpus.txt'
input_text = load_file(corpuspath)
# corpuspath = 'train.txt'
# input_text = load_file(corpuspath)


#%%
txtwv = [remove_list_tag(line.strip().split()) for line in input_text.split('\n') if line != '']
# standard_corpus = '/Users/Andy/Desktop/大二下/ERG 2050/ERG2050_Assignment7/data/Segmentor-master/pfr/corpus.txt'
#%%
import gensim
w2v = trainW2V(txtwv)
#%%
w2v.save('wordvector.bin')
#%%
txtnltk = []
for w in input_text.split('\n'):
    txtnltk.extend(remove_tag(s) for s in w.split())
freqdf = freq_func(txtnltk)
#%%
word2idx = dict((c,i) for c, i in zip(freqdf.word,freqdf.idx))
idx2word = dict((i,c) for c, i in zip(freqdf.word,freqdf.idx))
w2v = word2vec.Word2Vec.load('wordvector.bin')
#%%
init_weight_wv, idx2word, word2idx = initweightlist(w2v, idx2word, word2idx)
#%%
pickle.dump(word2idx, open('word2idx.pickle', 'wb'))
pickle.dump(idx2word, open('idx2word.pickle', 'wb'))

pickle.dump(init_weight_wv, open('init_weight_wv.pickle', 'wb'))
#%%
output_file = 'pfr.tagging.utf8'
standardcorpus = 'corpus.txt'
character_tagging(standardcorpus, output_file)
#%%
with open(output_file, encoding= 'utf-8') as f:
    lines =f.readlines()
    train_line = [[w[0] for w in line.split()] for line in lines]
    train_label = [w[2] for line in lines for w in line.split()]
    # cpprint(train_label[:10])

    # cpprint(train_line[:10])


np.shape(train_line)
#%%
from data_process import *
train_word_num = []
for line in train_line:
    train_word_num.extend(featContext(line, word2idx))

#%%
pickle.dump(train_word_num, open('train_word_num.pickle', 'wb'))
pickle.dump(train_label, open('train_label.pickle', 'wb'))
#%%