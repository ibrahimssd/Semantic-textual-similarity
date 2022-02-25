from scipy import stats
import logging
from torch import nn

def evaluate_test_set(model, data_loader, config_dict):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on test set")
    mse= nn.MSELoss()
    device = config_dict["device"]
    self_attention_config = config_dict['self_attention_config']
    train_loader , val_loader , test_loader = data_loader
    test_generator = iter(val_loader)
    num_batch=len(test_loader)
    y_true = list()
    y_pred = list()
    total_loss = 0
    
    #iterate over train batches
    for batch in range(5):
            
            try:
                # Samples the batch
                sent1_batch, sent2_batch, sent1_lengths, sent2_lengths,targets,raw_sent1,raw_sent2= next(test_generator)

            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                train_generator = iter(train_loader)
                sent1_batch, sent2_batch, sent1_lengths, sent2_lengths,targets,raw_sent1,raw_sent2= next(test_generator)
        
        
            sent1_batch = sent1_batch
            sent2_batch= sent2_batch
        
            predictions , attention_matrix = model.forward(sent1_batch, sent2_batch, sent1_lengths, sent2_lengths)
#             predictions= Variable(predictions)
#             targets = Variable(targets)        
            
            try:
                 batch_test_loss = mse(predictions,targets) 
                       
            except RuntimeError:
            
                raise Exception("nan values on regularization. Rremove regularization or add very small values")
            
            #save the loss for batches
            total_loss += batch_test_loss
            y_true+=list(targets)
            y_pred+=list(predictions)
    
    
    acc = r2_score(y_true, y_pred)
    print(' test_loss: %.3f, test_score: %.3f' %
                   (torch.mean(total_loss.data.float()), score))
#     return acc, torch.mean(total_loss.data.float())

    
    
    
