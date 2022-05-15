import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import similarity_score


"""
Wrapper class using Pytorch nn.Module to create the architecture for our model
Architecture is based on the paper: 
A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING
https://arxiv.org/pdf/1703.03130.pdf
"""


class SiameseBiLSTMAttention(nn.Module):
    def __init__(
        self,
        batch_size,
        output_size,
        hidden_size,
        vocab_size,
        embedding_size,
        embedding_weights,
        lstm_layers,
        device,
        bidirectional,
        self_attention_config,
        fc_hidden_size,
        n_classes,

    ):
        super(SiameseBiLSTMAttention, self).__init__()
        """
        Initializes model layers and loads pre-trained embeddings from task 1
        """
        # model hyper parameters
        self.batch_size = batch_size
        self.output_size = output_size
        self.lstm_hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lstm_layers = lstm_layers
        self.device = device
        self.bidirectional = bidirectional
        self.fc_hidden_size = fc_hidden_size
        self.num_classes = n_classes
        self.lstm_directions = (
            2 if self.bidirectional else 1
        )  # decide directions based on input flag

        # model layers
        # initialize the look-up table.
        self.embeddings = nn.Embedding(vocab_size, embedding_size)

        # assign the look-up table to the pre-trained fasttext word embeddings.
        self.embeddings.weight = nn.Parameter(embedding_weights)
        self.lookup_table = embedding_weights#self.embeddings

        # incase we are using bi-directional lstm we'd have to take care of bi-directional outputs in
        # subsequent layers.

        # initialize Bilstm layer
        self.bi_lstm = torch.nn.LSTM(self.embedding_size, self.lstm_hidden_size,
                                     lstm_layers, batch_first=True, bias=False, bidirectional=True)

        # initialize self attention layers
        self.SelfAtt = SelfAttention(2*self.lstm_hidden_size, self_attention_config['hidden_size'],
                                     self_attention_config['output_size'])

        # Initialize fully connected layer

        self.fc1 = nn.Linear(2*self.lstm_hidden_size *
                             self_attention_config['output_size'], self.fc_hidden_size)

        # Initialize fully connected layer for classification task
        self.fc2 = nn.Linear(batch_size, self.num_classes)

        # normalization and tanh layers
        self.normalize = nn.BatchNorm1d(batch_size)
        self.tanh = nn.Tanh()

        # initialize soft max layer
        self.sofmax = torch.nn.Softmax(dim=1)

        # cosine similarity
        self.cos = nn.CosineSimilarity()
        # add dropout layer with probability 0.8
        self.mask = torch.nn.Dropout(p=0.8)

    def init_hidden(self, batch_size):
        """
        Initializes hidden and context weight matrix before each
                forward pass through LSTM
        """
        h0 = torch.zeros(self.lstm_directions*self.lstm_layers,
                         batch_size, self.lstm_hidden_size)
        c0 = torch.zeros(self.lstm_directions*self.lstm_layers,
                         batch_size, self.lstm_hidden_size)

        h0 = Variable(h0, requires_grad=True)
        c0 = Variable(c0, requires_grad=True)

        return h0, c0

    def forward_once(self, batch, lengths):
        """
        Performs the forward pass for each batch
        """

        # batch shape: (batch_size, seq_len)
        batch_size, sequence_len = batch.size()
        # embeddings shape: ( batch_size, seq_len, embedding_size)

        #h_init,c_init = self.init_hidden(batch_size)
        input_batch_sequences = self.lookup_table[batch]
        input_batch_sequences = Variable(
            input_batch_sequences, requires_grad=True)

        output, (hn, cn) = self.bi_lstm(
            input_batch_sequences, (self.h_init, self.c_init))

        return output, (hn, cn)

    def forward(self, sent1_batch, sent2_batch, sent1_lengths, sent2_lengths, classification=False):
        """
        Performs the forward pass for each batch
        """

        # init context and hidden weights for lstm cell
        self.h_init, self.c_init = self.init_hidden(self.batch_size)
        output1, _ = self.forward_once(sent1_batch, sent1_lengths)
#         self.h_init,self.c_init = self.init_hidden(self.batch_size)
        output2, _ = self.forward_once(sent2_batch, sent2_lengths)

        # Self attention Layer
        attended_embeddings_sent1, attention_matrix_sent1 = self.SelfAtt.forward(
            output1)
        attended_embeddings_sent2, attention_matrix_sent2 = self.SelfAtt.forward(
            output2)

        # Fully connected layer

        final_embeddings_sent1 = self.tanh(self.normalize(
            self.fc1(attended_embeddings_sent1.reshape(output1.size(0), -1))))
        final_embeddings_sent2 = self.tanh(self.normalize(
            self.fc1(attended_embeddings_sent2.reshape(output1.size(0), -1))))

        # soft max layer  #decide whether a classification or regression task
        if classification:

            final_embeddings_sent1 = self.tanh(
                self.fc2(final_embeddings_sent1))
            final_embeddings_sent2 = self.tanh(
                self.fc2(final_embeddings_sent2))

            # prepare embeddings for classification
            pair_sentence_embeddings = final_embeddings_sent1+final_embeddings_sent2

            #soft_out = self.sofmax(pair_sentence_embeddings)
            # soft_out  # torch.max(soft_out,1)[1]

            # classification predictions for class lebels
            predictions = pair_sentence_embeddings
            attention_mat = torch.cat(
                (attention_matrix_sent1, attention_matrix_sent2), 2)

            return predictions, attention_mat

        else:

            # return similarity score prediction for target task (calculate cosine similarity score)

            # predictions = similarity_score(
            #     final_embeddings_sent1, final_embeddings_sent2)

            predictions = self.cos(
                final_embeddings_sent1, final_embeddings_sent2)
            attention_mat = torch.cat(
                (attention_matrix_sent1, attention_matrix_sent2), 2)

            return predictions, attention_mat


class SelfAttention(nn.Module):
    """
    Implementation of the attention block
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(SelfAttention, self).__init__()
        # TODO implement
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.ws1 = nn.Linear(input_size, hidden_size, bias=False)
        self.ws2 = nn.Linear(hidden_size, output_size, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    # the forward function would receive lstm's all hidden states as input
    def forward(self, attention_input):
        size = attention_input.size()
        inp = attention_input.reshape(size[0]*size[1], size[2])
        attention_matrix = self.softmax(self.ws2(self.tanh(self.ws1(inp))))
        attention_matrix = attention_matrix.reshape(
            size[0], self.output_size, -1)
        attended_embeddings_sent1 = torch.bmm(
            attention_matrix, attention_input)

        return attended_embeddings_sent1, attention_matrix
