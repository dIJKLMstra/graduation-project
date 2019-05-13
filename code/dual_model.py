'''
    @author Qi Sun
    @desc classify dual sentences using LSTM
'''
import os
import torch
import pickle

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import variable
from torch.utils.data import Dataset, DataLoader, TensorDataset

train_xdata = []
train_ydata = []
test_xdata = []
test_ydata = []
train_lens = []
test_lens = []
word_to_ix = {}
embed_dict = {}

# ARGUMENT SETTINGS
EMBEDDING_DIM = 300
HIDDEN_DIM = 100
CLASSIFY_DIM = 2
BATCH_SIZE = 32
SEQUENCE_LENGTH_MAX = 60    # the longest length of sequence using for padding

embedPath = '../cn_embed/sgns.merge.char'
trainPath = '../data/dataset/dual_train.tsv'
testPath = '../data/dataset/dual_test.tsv'
pickle_path = '../data/clf_pickles'
result_path = '../data/result'

def prepare_sequence(seq, to_ix):
    '''
        get the indices of a given sentence and return its tensor
    '''
    idxs = []
    for word in seq:
        vec = to_ix.get(word,[float(0)]*300)
        #vec = to_ix[word]
        vec = [float(v) for v in vec]
        idxs.append(vec)

    # padding
    while len(idxs) < SEQUENCE_LENGTH_MAX:
        idxs.append([float(0)]*300)
    return idxs
    #return torch.tensor(idxs, dtype=torch.long)

def save_embedding():
    '''
        save embedding of chinese characters
    '''

    with open(embedPath, 'r', encoding='utf-8') as embed:
        lines = embed.readlines()

    for line in lines:
        info = line[:-1].split()
        char = info[0]
        if len(char) != 1:
            continue
        vec = info[1:]
        embed_dict[char] = vec

    with open(os.path.join(pickle_path, \
        'embed_dict.pkl'), 'wb') as ed:
        pickle.dump(embed_dict, ed)

class dualDataset(Dataset):
    '''
        convenience for batching
    '''

    def __init__ (self, x_data, y_data, data_lens):
        self.x_data = torch.tensor(x_data, dtype=torch.float)
        self.y_data = torch.tensor(y_data, dtype=torch.long)
        self.data_lens = data_lens
        self.len = len(x_data)

    def __getitem__ (self, idx):
        return {'x_data': self.x_data[idx], \
            'y_data': self.y_data[idx], 'lens':self.data_lens[idx]}

    def __len__ (self):
        return self.len


def prepare_data():
    '''
        prepare the training sentences and label for LSTM
    '''
    global train_xdata
    global train_ydata
    global train_lens

    with open(trainPath, 'r', encoding='utf-8') as trainF:
        trainLines = trainF.readlines()

    with open(testPath, 'r', encoding='utf-8') as testF:
        testLines = testF.readlines()

    with open(os.path.join(pickle_path, \
        'embed_dict.pkl'), 'rb') as ed:
        embed_dict = pickle.load(ed)

    # dataLines = trainLines + testLines
    # for line in dataLines:
    #     info = line[:-1].split('\t')
    #     for word in info[0]:
    #         if word not in word_to_ix:
    #             word_to_ix[word] = len(word_to_ix)
    
    # turn the training and test sentences
    # into Tensors of word indices
    for line in trainLines:
        info = line[:-1].split('\t')
        idx_ts = prepare_sequence(info[0], embed_dict)
        train_xdata.append(idx_ts)
        train_ydata.append(int(info[1]))
        train_lens.append(len(info[0]))

    for line in testLines:
        info = line[:-1].split('\t')
        idx_ts = prepare_sequence(info[0], embed_dict)
        test_xdata.append(idx_ts)
        test_ydata.append(int(info[1]))
        test_lens.append(len(info[0]))

    with open(os.path.join(pickle_path, \
        'dual_train_xdata.pkl'), 'wb') as trainx:
        pickle.dump(train_xdata, trainx)

    with open(os.path.join(pickle_path, \
        'dual_train_ydata.pkl'), 'wb') as trainy:
        pickle.dump(train_ydata, trainy)

    with open(os.path.join(pickle_path, \
        'dual_train_len.pkl'), 'wb') as trainlen:
        pickle.dump(train_lens, trainlen)

    with open(os.path.join(pickle_path, \
        'dual_test_xdata.pkl'), 'wb') as testx:
        pickle.dump(test_xdata, testx)

    with open(os.path.join(pickle_path, \
        'dual_test_ydata.pkl'), 'wb') as testy:
        pickle.dump(test_ydata, testy)

    with open(os.path.join(pickle_path, \
        'dual_test_len.pkl'), 'wb') as testlen:
        pickle.dump(test_lens, testlen)

    # with open(os.path.join(pickle_path, \
    #     'word_to_ix.pkl'), 'wb') as wti:
    #     pickle.dump(word_to_ix, wti)

    

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence, sentence_lens):
        #embeds = self.word_embeddings(sentence)
        #lstm_out, lstm_hidden = self.lstm(embeds.view(len(sentence), 1, -1))
        embed_input_x_packed = nn.utils.rnn.pack_padded_sequence(\
            sentence, sentence_lens, batch_first=True, enforce_sorted=False)
        encoder_outputs_packed, (h_last, c_last) = self.lstm(embed_input_x_packed)
        #print(h_last.shape)
        tag_space = self.hidden2tag(h_last.view(h_last.shape[1], -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def train():
    '''
        training part
    '''
    global train_xdata
    global train_ydata
    global train_lens

    #train_xdata = torch.tensor(train_xdata, dtype=torch.float)
    #train_ydata = torch.tensor(train_ydata, dtype=torch.float)

    #deal_dataset = TensorDataset(train_xdata, train_ydata)
    #print(train_xdata[1])
    deal_dataset = dualDataset(train_xdata, train_ydata, train_lens)
    train_loader = DataLoader( \
        dataset=deal_dataset, batch_size=BATCH_SIZE)

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, CLASSIFY_DIM)
    # loss function
    loss_function = nn.NLLLoss()
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(100):  
        print(epoch)
        #for sentence_in, targets in training_data:
        for idx, data in enumerate(train_loader):

            sentences = data['x_data']
            targets = data['y_data']
            sent_lens = data['lens']

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Run our forward pass.
            tag_scores = model(sentences, sent_lens)

            # Step 3. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    with open(os.path.join(pickle_path, \
        'dual_model_baseline.pkl'), 'wb') as dm:
        pickle.dump(model, dm)

def model_pred():
    '''
        use the model we trained to give a prediction
    '''

    with open(os.path.join(pickle_path, \
        'dual_model_baseline.pkl'), 'rb') as dm:
        model = pickle.load(dm)

    with open(testPath, 'r', encoding='utf-8') as testF:
        testLines = testF.readlines()

    with open(os.path.join(pickle_path, \
        'dual_train_xdata.pkl'), 'rb') as trainx:
        train_xdata = pickle.load(trainx)

    with open(os.path.join(pickle_path, \
        'dual_train_ydata.pkl'), 'rb') as trainy:
        train_ydata = pickle.load(trainy)

    with open(os.path.join(pickle_path, \
        'dual_train_len.pkl'), 'rb') as trainlen:
        train_lens = pickle.load(trainlen)

    with open(os.path.join(pickle_path, \
        'dual_test_xdata.pkl'), 'rb') as testx:
        test_xdata = pickle.load(testx)
    
    with open(os.path.join(pickle_path, \
        'dual_test_ydata.pkl'), 'rb') as testy:
        test_ydata = pickle.load(testy)

    with open(os.path.join(pickle_path, \
        'dual_test_len.pkl'), 'rb') as testlen:
        test_lens = pickle.load(testlen)

    # with open(os.path.join(pickle_path, \
    #     'word_to_ix.pkl'), 'rb') as wti:
    #     word_to_ix = pickle.load(wti)

    deal_dataset1 = dualDataset(train_xdata, train_ydata, train_lens)
    train_loader = DataLoader( \
        dataset=deal_dataset1, batch_size=BATCH_SIZE)

    deal_dataset2 = dualDataset(test_xdata, test_ydata, test_lens)
    test_loader = DataLoader( \
        dataset=deal_dataset2, batch_size=BATCH_SIZE)

    resultFile_path = os.path.join(result_path,'dual_baseline_results.txt')
    with open(resultFile_path, 'w') as writeF:

        for loader in [train_loader, test_loader]:
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            total_cnt = 0
            for data in loader:
                sentences = data['x_data']
                targets = data['y_data']
                sent_lens = data['lens']
                
                prediction = model(sentences, sent_lens)
                prediction = torch.argmax(prediction, dim=1)
                target = targets.numpy()
                prediction = prediction.numpy()

                for idx in range(len(target)):
                    if  prediction[idx] == 0 and target[idx] == 0:
                        TN += 1
                    elif prediction[idx] == 0 and target[idx] == 1:
                        FN += 1
                    elif prediction[idx] == 1 and target[idx] == 0:
                        FP += 1
                    else:
                        TP += 1
                    total_cnt += 1

            writeF.write('-'*64 + '\n')
            writeF.write('correct_cnt: ' + str(TP + TN) + '\n')
            writeF.write('total_cnt: ' + str(total_cnt) + '\n')
            writeF.write('accuracy: ' + str(float((TP + TN) / total_cnt)) + '\n')
            writeF.write('precision: ' + str(float(TP / (TP + FP))) + '\n')
            writeF.write('recall: ' + str(float(TP/ (TP + FN))) + '\n')
            writeF.write('F1 score:' + str(float((2 * TP) / (2* TP + FP + FN))) + '\n')

if __name__ == "__main__":
    #prepare_data()
    #train()
    model_pred()
    #save_embedding()