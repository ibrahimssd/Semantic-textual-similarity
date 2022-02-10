import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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
    ):
        super(SiameseBiLSTMAttention, self).__init__()
        """
        Initializes model layers and loads pre-trained embeddings from task 1
        """
        ## model hyper parameters
        self.batch_size = batch_size
        self.output_size = output_size
        self.lstm_hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lstm_layers = lstm_layers
        self.device = device
        self.bidirectional = bidirectional
        self.fc_hidden_size = fc_hidden_size
        self.lstm_directions = (
            2 if self.bidirectional else 1
        )  ## decide directions based on input flag
        
        pass
        ## model layers
        # TODO initialize the look-up table.

        # TODO assign the look-up table to the pre-trained fasttext word embeddings.
        self.lookup=embedding_weights

        ## TODO initialize lstm layer
        self.bi_lstm = torch.nn.LSTM(self.embedding_size, self.lstm_hidden_size, 
                            lstm_layers, batch_first=True , bias= True,bidirectional=True)
        ## TODO initialize self attention layers
        self.atten = SelfAttention(self.lstm_hidden_size*2, self_attention_config.hidden_size, self_attention_config.output_size)
        ## incase we are using bi-directional lstm we'd have to take care of bi-directional outputs in
        ## subsequent layers


    def init_hidden(self, batch_size):
        """
        Initializes hidden and context weight matrix before each
                forward pass through LSTM
        """
        h0 = torch.randn(self.lstm_directions*self.lstm_layers, batch_size, self.lstm_hidden_size)
        c0 = torch.randn(self.lstm_directions*self.lstm_layers, batch_size, self.lstm_hidden_size)
        
        return h0, c0 

    def forward_once(self, batch, lengths):
        """
        Performs the forward pass for each batch
        """

        ## batch shape: (batch_size, seq_len)
        batch_size , sequence_len = batch.size()z
        ## embeddings shape: ( batch_size, seq_len, embedding_size)
        
#         h_init,c_init = self.init_hidden(batch_size)
        input_batch_sequences= self.lookup[batch]
        
        output, (hn, cn) = self.bi_lstm(input_batch_sequences, (self.h_init, self.c_init))

        return output , (hn , cn)

    def forward(self, sent1_batch, sent2_batch, sent1_lengths, sent2_lengths):
        """
        Performs the forward pass for each batch
        """
        ## TODO init context and hidden weights for lstm cell
        self.h_init,self.c_init = self.init_hidden(self.batch_size)
        output1,_ = self.forward_once(sent1_batch,sent1_lengths)
        att1 = self.atten(output1)
        self.h_init,self.c_init = self.init_hidden(self.batch_size)
        output2,_ = self.forward_once(sent2_batch,sent2_lengths)
        att2 = self.atten(output2)
        
        pass
        # TODO implement forward pass on both sentences. calculate similarity using similarity_score()


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
        
    ## the forward function would receive lstm's all hidden states as input
    def forward(self, attention_input):
        # TODO implement
        #pass
        size = attention_input.size()
        inp = attention_input.reshape(size[0]*size[1],size[2])
        a = self.softmax(self.ws2(self.tanh(self.ws1(inp))))
    
        a = a.reshape(size[0], self.output_size, -1)
        print("a:",a.shape)
        m = torch.bmm(a , attention_input)
        print("m:",m.shape)
        return a, m
                         
