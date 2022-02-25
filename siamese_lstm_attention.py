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
        
        
        
        ## model layers
        # TODO initialize the look-up table.
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        
        # TODO assign the look-up table to the pre-trained fasttext word embeddings.
        self.embeddings.weight = nn.Parameter(embedding_weights)
        self.lookup_table=  self.embeddings 
        
        ## TODO initialize lstm layer
        self.bi_lstm = torch.nn.LSTM(self.embedding_size, self.lstm_hidden_size, 
                            lstm_layers, batch_first=True , bias= True,bidirectional=True)
        
        ## TODO initialize self attention layers
        self.SelfAtt= SelfAttention(2*self.lstm_hidden_size, self_attention_config['hidden_size'],
                                    self_attention_config['output_size'])
        
        #Initialize fully connected layer
        self.fc = nn.Linear(2*self.lstm_hidden_size * self_attention_config['output_size'], self.fc_hidden_size )
        self.tanh = nn.Tanh()
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
        batch_size , sequence_len = batch.size()
        ## embeddings shape: ( batch_size, seq_len, embedding_size)
        
        #h_init,c_init = self.init_hidden(batch_size)
        input_batch_sequences= self.lookup_table(batch)
        
        output, (hn, cn) = self.bi_lstm(input_batch_sequences, (self.h_init, self.c_init))

        return output , (hn , cn)

    def forward(self, sent1_batch, sent2_batch, sent1_lengths, sent2_lengths):
        """
        Performs the forward pass for each batch
        """
        
        
        ## init context and hidden weights for lstm cell
        self.h_init,self.c_init = self.init_hidden(self.batch_size)
        output1,_ = self.forward_once(sent1_batch,sent1_lengths)
        self.h_init,self.c_init = self.init_hidden(self.batch_size)
        output2,_ = self.forward_once(sent2_batch,sent2_lengths)
        
        ## Self attention Layer
        attended_embeddings_sent1, attention_matrix_sent1 = self.SelfAtt.forward(output1)
        attended_embeddings_sent2, attention_matrix_sent2 = self.SelfAtt.forward(output2)
        
        ## Fully connected layer 
        
        final_embeddings_sent1= self.tanh(self.fc(attended_embeddings_sent1.reshape(output1.size(0),-1)))
        final_embeddings_sent2= self.tanh(self.fc(attended_embeddings_sent2.reshape(output2.size(0),-1)))
        
        
        #similarity score prediction
        predictions = similarity_score(final_embeddings_sent1, final_embeddings_sent2)
        
        
        return predictions , torch.cat((attention_matrix_sent1, attention_matrix_sent2), 1)

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
        attention_matrix = self.softmax(self.ws2(self.tanh(self.ws1(inp))))
        attention_matrix= attention_matrix.reshape(size[0], self.output_size, -1)
        attended_embeddings_sent1= torch.bmm(attention_matrix , attention_input)
        
        return attended_embeddings_sent1, attention_matrix