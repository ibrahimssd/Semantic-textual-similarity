import torch
from torch import nn
import logging
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.metrics import r2_score
from utils import plot_stats
logging.basicConfig(level=logging.INFO)

"""
Script for training the neural network and saving the better models 
while monitoring a metric like accuracy etc
"""


def train_model(model, optimizer, dataloader, data, max_epochs, config_dict):
    device = config_dict["device"]
    self_attention_config = config_dict['self_attention_config']
    criterion = nn.MSELoss()
    max_score = -1.7976931348623157e+308 #5e-1
    train_loader , val_loader , test_loader = dataloader
    train_generator = iter(train_loader)
    num_batch=len(train_loader)
    
    #save information for visualization
    dictionary_info={}
    dictionary_info['train_loss']=[]
    dictionary_info['val_loss']=[]
    dictionary_info['epochs']=[]
    for epoch in tqdm(range(max_epochs)):
        y_true = list()
        y_pred = list()
        total_loss=0
        
        #iterate over train batches
        for batch in range(1):
            
            try:
                # Samples the batch
                sent1_batch, sent2_batch, sent1_lengths, sent2_lengths,targets,raw_sent1,raw_sent2= next(train_generator)

            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                train_generator = iter(train_loader)
                sent1_batch, sent2_batch, sent1_lengths, sent2_lengths,targets,raw_sent1,raw_sent2= next(train_generator)
        
        
            sent1_batch = Variable(sent1_batch)
            sent2_batch= Variable(sent2_batch)
        
            predictions , attention_matrix = model.forward(sent1_batch, sent2_batch, sent1_lengths, sent2_lengths)
            predictions= Variable(predictions)
            targets = Variable(targets)
            
            
            
        
        
        
            try:
                 
                batch_train_loss = criterion(predictions,targets) + (attention_penalty_loss(attention_matrix, 
                                                                  self_attention_config['penalty'], device))
                       
            except RuntimeError:
            
                raise Exception("nan values on regularization. Rremove regularization or add very small values")
            
            
            
            batch_train_loss= Variable(batch_train_loss)
            batch_train_loss.requires_grad=True
            optimizer.zero_grad(set_to_none=True) #or  model.zero_grad()
            batch_train_loss.backward()
            optimizer.step()
            total_loss += batch_train_loss
            y_true+=list(targets)
            y_pred+=list(predictions)
            
        
            
        
        
        
        
        # TODO: computing accuracy using sklearn's function
        
        #accuracy = (torch.argmax(predictions, axis=-1) == targets).float().mean()
        score = r2_score(y_true, y_pred)
        
        
        ## compute model metrics on dev set
        val_score, val_loss = evaluate_dev_set(
            model, data, criterion, val_loader, config_dict, device
        )

        
        
        if val_score > max_score:
            max_score = val_score
            logging.info(
                "new model saved")  
            
            ## save the model if it is better than the prior best
            torch.save(model.state_dict(), "{}.pth".format(config_dict["model_name"]))

        
        logging.info(
            "Train loss: {} - Train score: {} -- Validation loss: {} - Validation score: {}".format(
                torch.mean(total_loss.data.float()), score, val_loss, val_score
            )
        )

        

        if True: #epoch % 100 == 0:
             print('[%d/%d] train_loss: %.3f, accuracy_score: %.3f' %
                   (epoch , max_epochs - 1,torch.mean(total_loss.data.float()), score))
        if epoch == max_epochs - 1:
             print('Final score: %.3f, expected %.3f' %
                         (score, 1.0))
        
    
        dictionary_info['train_loss'].append(total_loss.data.float().item())
        dictionary_info['val_loss'].append(val_loss.item())
        dictionary_info['epochs'].append(epoch)
    #Visualize the progress
    
    plot_stats(dictionary_info)
    return model


def evaluate_dev_set(model, data, criterion, data_loader, config_dict, device):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on dev set")
    
    
    device = config_dict["device"]
    self_attention_config = config_dict['self_attention_config']
    val_loader = data_loader
    val_generator = iter(val_loader)
    num_batch=len(val_loader)
    y_true = list()
    y_pred = list()
    total_loss = 0
    
    #iterate over train batches
    for batch in range(1):
            
            try:
                # Samples the batch
                sent1_batch, sent2_batch, sent1_lengths, sent2_lengths,targets,raw_sent1,raw_sent2= next(val_generator)

            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                train_generator = iter(train_loader)
                sent1_batch, sent2_batch, sent1_lengths, sent2_lengths,targets,raw_sent1,raw_sent2= next(val_generator)
        
        
            sent1_batch = Variable(sent1_batch)
            sent2_batch= Variable(sent2_batch)
        
            predictions , attention_matrix = model.forward(sent1_batch, sent2_batch, sent1_lengths, sent2_lengths)
#             predictions= Variable(predictions
#             targets = Variable(targets)        
            
            try:
                 batch_val_loss = criterion(predictions,targets) + (attention_penalty_loss(attention_matrix, 
                                                                  self_attention_config['penalty'], device))
                       
            except RuntimeError:
            
                raise Exception("nan values on regularization. Rremove regularization or add very small values")
            
            #save the loss for batches
            total_loss += batch_val_loss
            y_true+=list(targets.detach().numpy())
            y_pred+=list(predictions.detach().numpy())
    
    
    acc = r2_score(y_true, y_pred)
    
    return acc, torch.mean(total_loss.data.float())

def attention_penalty_loss(annotation_weight_matrix, penalty_coef, device):
    """
    This function computes the loss from annotation/attention matrix
    to reduce redundancy in annotation matrix and for attention
    to focus on different parts of the sequence corresponding to the
    penalty term 'P' in the ICLR paper
    ----------------------------------
    'annotation_weight_matrix' refers to matrix 'A' in the ICLR paper
    annotation_weight_matrix shape: (batch_size, attention_out, seq_len)
    """
    
    batch_size, attention_out_size = annotation_weight_matrix.size(0), annotation_weight_matrix.size(1)
    annotation_weight_matrix_trans = annotation_weight_matrix.transpose(1,2)
    identity = torch.eye(attention_out_size)
    identity = Variable(identity.unsqueeze(0).expand(batch_size,attention_out_size,attention_out_size))
    annotation_mul_difference=annotation_weight_matrix@annotation_weight_matrix_trans - identity
    penalty = frobenius_norm(annotation_mul_difference)
    return penalty_coef*penalty


def frobenius_norm(annotation_mul_difference):
    """
    Computes the frobenius norm of the annotation_mul_difference input as matrix
    """
   
    """
    
 
    Args:
      annotation_mul_difference= ||AAT - I||
 
    Returns:
            regularized value
 
       
        """
    
#    torch.norm(annotation_mul_difference.float(), p='fro')
#     torch.sum(torch.sum(torch.sum(annotation_mul_difference**2,1),1)**0.5).type(torch.DoubleTensor)
    return torch.sqrt(torch.sum(annotation_mul_difference**2))