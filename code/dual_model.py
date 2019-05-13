import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


training_data = []
test_data = []
word_to_ix = {}
# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 100
HIDDEN_DIM = 100

def prepare_sequence(seq, to_ix):
    '''
        get the indices of a given sentence and return its tensor
    '''

    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prepare_data():
    '''
        prepare the training sentences and label for LSTM
    '''

    trainPath = '../data/dataset/dual_train.tsv'
    testPath = '../data/dataset/dual_test.tsv'

    with open(trainPath, 'r', encoding='utf-8') as trainF:
        trainLines = trainF.readlines()

    with open(testPath, 'r', encoding='utf-8') as testF:
        testLines = testF.readlines()

    dataLines = trainLines + testLines
    for line in dataLines:
        info = line[:-1].split('\t')
        for word in info[0]:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    
    for line in trainLines:
        info = line[:-1].split('\t')
        idx_ts = prepare_sequence(info[0], word_to_ix)
        label = torch.tensor([int(info[1])], dtype=torch.long)
        training_data.append((idx_ts, label))

    for line in testLines:
        info = line[:-1].split('\t')
        idx_ts = prepare_sequence(info[0], word_to_ix)
        label = torch.tensor([int(info[1])], dtype=torch.long)
        test_data.append((idx_ts, label))

    print(training_data)
    

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, lstm_hidden = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out[-1].view(1, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def train():
	'''
		training part
	'''

	model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), 2)
	loss_function = nn.NLLLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.1)

	# See what the scores are before training
	# Note that element i,j of the output is the score for tag j for word i.
	# Here we don't need to train, so the code is wrapped in torch.no_grad()
	with torch.no_grad():
	    inputs = training_data[0][0]
	    tag_scores = model(inputs)
	    print(tag_scores)

	for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
	    for sentence_in, targets in training_data:
	        # Step 1. Remember that Pytorch accumulates gradients.
	        # We need to clear them out before each instance
	        model.zero_grad()

	        # Step 2. Get our inputs ready for the network, that is, turn them into
	        # Tensors of word indices.
	        #sentence_in = prepare_sequence(sentence, word_to_ix)
	        #targets = prepare_sequence(tags, tag_to_ix)

	        # Step 3. Run our forward pass.
	        tag_scores = model(sentence_in)

	        # Step 4. Compute the loss, gradients, and update the parameters by
	        #  calling optimizer.step()
	        loss = loss_function(tag_scores, targets)
	        loss.backward()
	        optimizer.step()

	# See what the scores are after training
	with torch.no_grad():
	    inputs = training_data[0][0]
	    tag_scores = model(inputs)

	    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
	    # for word i. The predicted tag is the maximum scoring tag.
	    # Here, we can see the predicted sequence below is 0 1 2 0 1
	    # since 0 is index of the maximum value of row 1,
	    # 1 is the index of maximum value of row 2, etc.
	    # Which is DET NOUN VERB DET NOUN, the correct sequence!
	    print(tag_scores)

if __name__ == "__main__":
	prepare_data()
	train()