import os
import numpy as np
from lstmNet import *

class NN(object):  # LSTM, RNN
    def __init__(self, train_data, test_data, test_label):
        self.train = train_data
        self.test = test_data
        self.label = test_label        

    def TRAIN(self):
        train_word_num = pickle.load(open('train_word_num.pickle', 'rb'))
        train_label = pickle.load(open('train_label.pickle', 'rb'))
        nb_classes = len(np.unique(train_label))
        init_weight_wv = pickle.load(open('init_weight_wv.pickle', 'rb'))
        label_dict = dict(zip(np.unique(train_label), range(4)))
        num_dict = {n:l for l, n in label_dict.items()}
        train_label = [label_dict[y] for y in train_label]
        train_word_num = np.array(train_word_num)
        #stacking LSTM
        modelname = 'my_model_weights.h5'
        net = Lstm_Net()
        net.init_weight = [np.array(init_weight_wv)]
        net.nb_classes = nb_classes
        net.splitset(train_word_num,train_label)
        print('training...')
        net.buildnet()
        net.train(modelname)

    def TEST(self):
        word2idx = pickle.load(open('word2idx.pickle','rb'))
        train_word_num = pickle.load(open('train_word_num.pickle', 'rb'))
        train_label = pickle.load(open('train_label.pickle', 'rb'))
        nb_classes = len(np.unique(train_label))
        
        init_weight_wv = pickle.load(open('init_weight_wv.pickle', 'rb'))
        
        label_dict = dict(zip(np.unique(train_label), range(4)))
        num_dict = {n:l for l,n in label_dict.items()}
        
        net = Lstm_Net()
        net.init_weight = [np.array(init_weight_wv)]
        net.nb_classes = nb_classes
        net.buildnet()
        net.getweights('my_model_weights.h5')
        test_data = self.test_data      
        pred = []
        with open('my_prediction.txt','w',encoding="utf-8") as f:
            f.close()                                       
        for temp_txt in test_data:
            temp_txt = list(temp_txt)
            temp_num = featContext(temp_txt,word2idx = word2idx)
            temp = net.predict_num(temp_num, temp_txt, label_dict = label_dict, num_dict = num_dict)
            with open('my_prediction.txt','a',encoding="utf-8") as f:
                f.write(normalPrint(temp))
                f.write("\n")

def main():
    train_data = open(os.path.join('data', 'train.txt')).read().splitlines()
    test_data = open(os.path.join('data', 'test.txt')).read().splitlines()
    test_label = open(os.path.join('data', 'test_gold.txt')).read().splitlines()
    model = NN(train_data, test_data, test_label)
    print(1)
    model.TRAIN()
    model.TEST()

    # Write your prediction to a file (the format should be the same as test_gold.txt)
    # You need to convert BIES to the format like test_gold.txt (call hmm.word_segmentation() here)
    fout = open('my_prediction.txt', 'w', encoding='utf-8')
    my_pred = fout.read()
    count_index = 0
    pred_count = 0
    total_count = 0
    my_pred = my_pred.split("\n")
    test_gold = test_data_gold
    for num in range(len(my_pred)):
        sentence_divide = my_pred[num].split("  ")
        sentence_divide_perfect = test_gold[num].split("  ")
        pos = 0
        pred_pos = []
        good_pos = []
        for word in sentence_divide:
            pos_prev = pos
            if len(word) == 1 or len(word)==0:
                pos_latter = pos
            else:
                pos_latter = pos + len(word)-1
            pos = pos_latter + 1
            pred_pos.append((pos_prev,pos_latter))
        pos = 0
        for word in sentence_divide_perfect:
            pos_prev = pos
            if len(word) == 1:
                pos_latter = pos
            else:
                pos_latter = pos + len(word)-1
            pos = pos_latter + 1
            good_pos.append((pos_prev,pos_latter))
        for index in pred_pos:
            if index in good_pos:
                count_index += 1
        pred_count += len(pred_pos)
        total_count += len(good_pos)
    precision = count_index / pred_count
    recall = count_index / total_count
    f1 = (2*precision*recall) / (precision + recall)
    print("The precision is: %.2f%%, The recall is: %.2f%%, the f1 score is: %.5f."%(precision*100, recall*100, f1))

main()

