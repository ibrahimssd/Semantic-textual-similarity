import torch
from scipy import stats
import logging
from torch import nn
from sklearn.metrics import r2_score
from torch.autograd import Variable
from train import attention_penalty_loss

def evaluate_test_set(model, data_loader, config_dict):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating score on test set")
    
    mse= nn.MSELoss()
    device = config_dict["device"]
    self_attention_config = config_dict['self_attention_config']
    train_loader , val_loader , test_loader = data_loader
    test_generator = iter(test_loader)
    num_test_batch=len(test_loader)
    y_true = list()
    y_pred = list()
    total_loss = 0
    
    #iterate over test batches
    for batch in range(num_test_batch):
            
            # protect dataloader sampling process
            try:
                #  batch sampling 
                sent1_batch, sent2_batch, sent1_lengths, sent2_lengths,targets,raw_sent1,raw_sent2= next(test_generator)

            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                train_generator = iter(train_loader)
                sent1_batch, sent2_batch, sent1_lengths, sent2_lengths,targets,raw_sent1,raw_sent2= next(test_generator)                        
            predictions , attention_matrix = model.forward(sent1_batch, sent2_batch, sent1_lengths, sent2_lengths)
            
            # protect loss calculations from nan values 
            try:
                    batch_test_loss = mse(predictions,targets) + attention_penalty_loss(attention_matrix,                                                                                      self_attention_config['penalty'], device)         
                    
            except RuntimeError:
                  
                    raise Exception("nan values on regularization. Rremove regularization or add very small values")
            
            #save the loss for test batches
            total_loss += batch_test_loss
            y_true+=list(targets.detach().numpy())
            y_pred+=list(predictions.detach().numpy())
    
    
    test_score = r2_score(y_true, y_pred)
    
    print(' test_loss: %.3f, test_score: %.3f' % (total_loss.data.float()/num_test_batch, test_score))


    
    
    
