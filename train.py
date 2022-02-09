import torch
from torch import nn
import logging
from tqdm import tqdm
from torch.autograd import Variable


logging.basicConfig(level=logging.INFO)

"""
Script for training the neural network and saving the better models 
while monitoring a metric like accuracy etc
"""


def train_model(model, optimizer, dataloader, data, max_epochs, config_dict):
    device = config_dict["device"]
    criterion = nn.MSELoss()
    max_accuracy = 5e-1
    train_loader , val_loader , test_loader = dataloader
    train_generator = iter(train_loader)
    
    dictionary_info={}
    for epoch in tqdm(range(max_epochs)):
        
        try:
            # Samples the batch
            sent1_batch, sent2_batch, sent1_lengths, sent2_lengths,targets,raw_sent1,raw_sent2= next(train_generator)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            generator = iter(trainloader)
            sent1_batch, sent2_batch, sent1_lengths, sent2_lengths, targets,raw_sent1,raw_sent2= next(train_generator)
        
        
        predictions= model.forward(sent1_batch, sent2_batch, sent1_lengths, sent2_lengths)
        optimizer.zero_grad()
        loss = criterion(predictions,targets) + attention_penalty_loss(annotation_weight_matrix, 
                                                                  penalty_coef, device)
        optimizer.step()
        
        # TODO: computing accuracy using sklearn's function
        ## acc = 
        #accuracy = (torch.argmax(predictions, axis=-1) == targets).float().mean()
        acc=accuracy_score(y_true, y_pred)

        ## compute model metrics on dev set
        val_acc, val_loss = evaluate_dev_set(
            model, data, criterion, dataloader, config_dict, device
        )

        
        
        if val_acc > max_accuracy:
            max_accuracy = val_acc
            logging.info(
                "new model saved")  
            
            ## save the model if it is better than the prior best
            torch.save(model.state_dict(), "{}.pth".format(config_dict["model_name"]))

        logging.info(
            "Train loss: {} - acc: {} -- Validation loss: {} - acc: {}".format(
                torch.mean(total_loss.data.float()), acc, val_loss, val_acc
            )
        )
        
        if epoch % 100 == 0:
        print('[%d/%d] loss: %.3f, accuracy: %.3f' %
                   (i , max_epochs - 1, loss.item(), acc.item()))
        if epoch == max_epochs - 1:
               print('Final accuracy: %.3f, expected %.3f' %
                         (accuracy.item(), 1.0))
        
    return model


def evaluate_dev_set(model, data, criterion, data_loader, config_dict, device):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on dev set")

    # TODO implement
    pass

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
    annotation_weight_matrix_trans = torch.transpose(annotation_weight_matrix, 0, 1)
    identity = torch.eye(annotation_weight_matrix.size(0))
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