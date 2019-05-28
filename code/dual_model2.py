import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

training_data = []
dev_data = []
test_data = []
# dictionary used for saving character level embeddings
embed_dict = {}

# parameters setting
EMBEDDING_DIM = 601
HIDDEN_DIM = 100
SEQUENCE_LENGTH_MAX = 25    # the longest length of sequence using for padding
PREDICTION_SIZE = 2

pickle_path = '../data/clf_pickles'
result_path = '../data/result'


def prepare_sequence(word1, word2, to_vec):
    '''
        get the embedding of a given sentence and return its tensor
    '''

    vec1 = to_vec.get(word1, [float(0)]*300)
    vec2 = to_vec.get(word2, [float(0)]*300)
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    vec_dot = np.dot(vec1, vec2)
    vec = np.append(vec1, vec2)
    vec = np.append(vec, vec_dot)
    return vec

def fill_datalist(lines, data_list):
    '''
        get sentences' embedding and label in different dataset
    '''

    for line in lines:
        info = line[:-1].split('\t')
        if len(info[0]) != len(info[1]):
            continue
        idx_ts = [prepare_sequence(info[0][idx], info[1][idx], embed_dict)\
                  for idx in range(len(info[0]))]
        idx_ts = torch.tensor(idx_ts, dtype=torch.float)
        label = torch.tensor([int(info[-1])], dtype=torch.long)
        data_list.append([idx_ts, label])

def prepare_data():
    '''
        prepare the training sentences and label for LSTM
    '''

    global embed_dict

    trainPath = '../data/processing_data/dual_train_divided2.tsv'
    devPath = '../data/processing_data/dual_dev_divided2.tsv'
    testPath = '../data/processing_data/dual_test_divided2.tsv'

    with open(trainPath, 'r', encoding='utf-8') as trainF:
        trainLines = trainF.readlines()

    with open(trainPath, 'r', encoding='utf-8') as devF:
        devLines = devF.readlines()

    with open(testPath, 'r', encoding='utf-8') as testF:
        testLines = testF.readlines()

    with open(os.path.join(pickle_path, \
        'embed_dict.pkl'), 'rb') as ed:
        embed_dict = pickle.load(ed)
    
    fill_datalist(trainLines, training_data)
    fill_datalist(devLines, dev_data)
    fill_datalist(testLines, test_data)
   
    
class dualLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, sentence_size, pred_size):
        super(dualLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = 0.5
        self.layer_size = 2
        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.

        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, self.layer_size,\
                            dropout=self.dropout_rate, bidirectional=True)
        #self.lstm2 = nn.LSTM(embedding_dim, hidden_dim)
        #self.lstm3 = nn.LSTM(2*hidden_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.vec2pred = nn.Linear(2 * self.layer_size * hidden_dim, pred_size)
        # set a dropout layer for linear layer
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, embeds):
        #embeds1 = self.word_embeddings(sentence1)
        #embeds2 = self.word_embeddings(sentence2)

        #packed_input1 = pack_padded_sequence(embeds1, sent_len)
        #packed_output1, (ht1, ct1) = self.lstm1(packed_input1)
        #output1, _ = pad_packed_sequence(packed_output1)

        #packed_input2 = pack_padded_sequence(embeds2, sent_len)
        #packed_output2, (ht2, ct2) = self.lstm1(packed_input1)
        #output2, _ = pad_packed_sequence(packed_output2)

        # vec = [torch.mm(lstm_hidden, output2[hidx].t()) \
        #     for hidx, lstm_hidden in enumerate(output1)]
        # vec = torch.stack(vec)
        
        #vec = [torch.cat((output1[i], output2[i]),0) \
        #    for i in range(sent_len[0])]
        #vec = torch.stack(vec).view(sent_len[0], 1, -1)
        #lstm_out, last_hidden = self.lstm3(vec)
        lstm_out, last_hidden = self.lstm1(embeds)
        pred_space = self.vec2pred(last_hidden[0].view(1, -1))
        pred_space = self.dropout(pred_space)
        prediction = F.log_softmax(pred_space, dim=1)

        return prediction

def train():
    '''
        training part
    '''

    model = dualLSTM(EMBEDDING_DIM, HIDDEN_DIM, \
        SEQUENCE_LENGTH_MAX, PREDICTION_SIZE)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    dev_accuracy = 0
    for epoch in range(10):  
        print('*****epoch ',epoch, '******')
        total_loss = 0
        model.train()
        for idx,data in enumerate(training_data):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. pass our data into our model
            data[0] = data[0].view(data[0].shape[0], 1, -1)
            #data[1] = data[1].view(SEQUENCE_LENGTH_MAX, 1, -1)
            prediction = model(data[0])

            # Step 3. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(prediction, data[1])
            total_loss += loss
            loss.backward()
            optimizer.step()
        # print out loss value so that we can know how many epoch we need
        total_loss = float(total_loss/ len(training_data))
        print('loss: ', total_loss)

        ## then we try to save the model with best performance in dev set
        model.eval()
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        total_cnt = 0
        for data in dev_data:
            data[0] = data[0].view(data[0].shape[0], 1, -1)
            prediction = model(data[0])
            prediction = torch.argmax(prediction, dim=1)
            zero_tensor = torch.tensor([0], dtype=torch.long)
            one_tensor = torch.tensor([1], dtype=torch.long)
            if  prediction == zero_tensor and data[1] == zero_tensor:
                TN += 1
            elif prediction == zero_tensor and data[1] == one_tensor:
                FN += 1
            elif prediction == one_tensor and data[1] == zero_tensor:
                FP += 1
            else:
                TP += 1
            total_cnt += 1
        accuracy = float((TP + TN) / total_cnt)
        print('accuracy: ', accuracy)
        if accuracy > dev_accuracy:
            dev_accuracy = accuracy
            with open(os.path.join(pickle_path, \
                'dual_model_lstm_dropout.pkl'), 'wb') as dm:
                pickle.dump(model, dm)

    with open(os.path.join(pickle_path, \
        'dual_model_lstm_dropout.pkl'), 'rb') as dm:
        model = pickle.load(dm)

    model.eval()
    resultFile_path = os.path.join(result_path,'dual_lstm_dropout_results.txt')
    with open(resultFile_path, 'w') as writeF:
        for dataset in [training_data, test_data]:
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            total_cnt = 0
            for data in dataset:
                data[0] = data[0].view(data[0].shape[0], 1, -1)
                #data[1] = data[1].view(SEQUENCE_LENGTH_MAX, 1, -1)
                prediction = model(data[0])
                prediction = torch.argmax(prediction, dim=1)
                #target = targets.numpy()
                #prediction = prediction.numpy()
                zero_tensor = torch.tensor([0], dtype=torch.long)
                one_tensor = torch.tensor([1], dtype=torch.long)
                #for idx in range(len(target)):
                if  prediction == zero_tensor and data[1] == zero_tensor:
                    TN += 1
                elif prediction == zero_tensor and data[1] == one_tensor:
                    FN += 1
                elif prediction == one_tensor and data[1] == zero_tensor:
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
    prepare_data()
    train()