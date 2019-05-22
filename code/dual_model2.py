import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

training_data = []
test_data = []
word_to_ix = {}
embed_dict = {}
# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 300
HIDDEN_DIM = 100
SEQUENCE_LENGTH_MAX = 25    # the longest length of sequence using for padding
PREDICTION_SIZE = 2

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
    return torch.tensor(idxs, dtype=torch.float)

def prepare_data():
    '''
        prepare the training sentences and label for LSTM
    '''
    global embed_dict

    trainPath = '../data/processing_data/dual_train_divided.tsv'
    testPath = '../data/processing_data/dual_test_divided.tsv'

    with open(trainPath, 'r', encoding='utf-8') as trainF:
        trainLines = trainF.readlines()

    with open(testPath, 'r', encoding='utf-8') as testF:
        testLines = testF.readlines()

    with open(os.path.join(pickle_path, \
        'embed_dict.pkl'), 'rb') as ed:
        embed_dict = pickle.load(ed)

    # dataLines = trainLines + testLines
    # for line in dataLines:
    #     info = line.split('\t')
    #     for word in info[0] + info[1]:
    #         if word not in word_to_ix:
    #             word_to_ix[word] = len(word_to_ix)
    
    for line in trainLines:
        info = line[:-1].split('\t')
        if len(info[0]) != len(info[1]):
            continue
        idx_ts1 = prepare_sequence(info[0], embed_dict)
        idx_ts2 = prepare_sequence(info[1], embed_dict)
        label = torch.tensor([int(info[-1])], dtype=torch.long)
        training_data.append([idx_ts1, idx_ts2, len(info[0]), label])

    for line in testLines:
        info = line[:-1].split('\t')
        if len(info[0]) != len(info[1]):
            continue
        idx_ts1 = prepare_sequence(info[0], embed_dict)
        idx_ts2 = prepare_sequence(info[1], embed_dict)
        label = torch.tensor([int(info[-1])], dtype=torch.long)
        test_data.append([idx_ts1, idx_ts2, len(info[0]), label])

    #print(test_data)
    
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, sentence_size, pred_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm2 = nn.LSTM(embedding_dim, hidden_dim)

        self.lstm3 = nn.LSTM(2*hidden_dim, hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.vec2pred = nn.Linear(hidden_dim, pred_size)

    def forward(self, embeds1, embeds2, sent_len):
        #embeds1 = self.word_embeddings(sentence1)
        #embeds2 = self.word_embeddings(sentence2)

        packed_input1 = pack_padded_sequence(embeds1, sent_len)
        packed_output1, (ht1, ct1) = self.lstm1(packed_input1)
        output1, _ = pad_packed_sequence(packed_output1)

        packed_input2 = pack_padded_sequence(embeds2, sent_len)
        packed_output2, (ht2, ct2) = self.lstm1(packed_input1)
        output2, _ = pad_packed_sequence(packed_output2)

        # vec = [torch.mm(lstm_hidden, output2[hidx].t()) \
        #     for hidx, lstm_hidden in enumerate(output1)]
        # vec = torch.stack(vec)
        
        vec = [torch.cat((output1[i], output2[i]),0) \
            for i in range(sent_len[0])]
        vec = torch.stack(vec).view(sent_len[0], 1, -1)
        lstm_out, last_hidden = self.lstm3(vec)
        pred_space = self.vec2pred(last_hidden[0].view(1, -1))
        prediction = F.log_softmax(pred_space, dim=1)

        return prediction

def train():
    '''
        training part
    '''

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, \
        SEQUENCE_LENGTH_MAX, PREDICTION_SIZE)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.03)

    for epoch in range(50):  # again, normally you would NOT do 300 epochs, it is toy data
        print(epoch)
        total_loss = 0
        for idx,data in enumerate(training_data):

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Run our forward pass.
            data[0] = data[0].view(SEQUENCE_LENGTH_MAX, 1, -1)
            data[1] = data[1].view(SEQUENCE_LENGTH_MAX, 1, -1)
            prediction = model(data[0], data[1], [data[2]])

            # Step 3. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(prediction, data[3])
            total_loss += loss
            loss.backward()
            optimizer.step()
        total_loss = float(total_loss/ len(training_data))
        print(total_loss)

    with open(os.path.join(pickle_path, \
        'dual_model_2lstm&dot.pkl'), 'wb') as dm:
        pickle.dump(model, dm)

    resultFile_path = os.path.join(result_path,'dual_2lstm&dot_results.txt')
    with open(resultFile_path, 'w') as writeF:
        for dataset in [training_data, test_data]:
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            total_cnt = 0
            for data in dataset:
                if len(data[0]) != len(data[1]):
                    continue
                data[0] = data[0].view(SEQUENCE_LENGTH_MAX, 1, -1)
                data[1] = data[1].view(SEQUENCE_LENGTH_MAX, 1, -1)
                prediction = model(data[0], data[1], [data[2]])
                prediction = torch.argmax(prediction, dim=1)
                #target = targets.numpy()
                #prediction = prediction.numpy()
                zero_tensor = torch.tensor([0], dtype=torch.long)
                one_tensor = torch.tensor([1], dtype=torch.long)
                #for idx in range(len(target)):
                if  prediction == zero_tensor and data[3] == zero_tensor:
                    TN += 1
                elif prediction == zero_tensor and data[3] == one_tensor:
                    FN += 1
                elif prediction == one_tensor and data[3] == zero_tensor:
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