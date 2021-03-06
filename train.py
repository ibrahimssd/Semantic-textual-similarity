import torch
from torch import nn
import logging
import numpy as np

from tqdm import tqdm
from torch.autograd import Variable
from scipy.stats import spearmanr
from utils import plot_progress
from sklearn.metrics import r2_score
from datetime import datetime
logging.basicConfig(level=logging.INFO)

"""
Script for training the neural network and saving the better models 
while monitoring a metric like accuracy etc
"""

# classification training functions


def train_classifier_model(model, optimizer, scheduler, dataloader, max_epochs, config_dict, clip=False, classify=False):

    device = config_dict["device"]
    self_attention_config = config_dict['self_attention_config']
    loss = nn.CrossEntropyLoss()
    train_loader, _, _ = dataloader
    train_generator = iter(train_loader)
    batch_train_num = len(train_loader)

    for epoch in tqdm(range(max_epochs)):
        y_true = list()
        y_pred = list()
        total_loss = 0
        correct = 0

        # iterate over train batches
        for batch in range(batch_train_num):

            # protection block for restarting sampling in case dataloader stops
            try:
                # batch Sampling
                sent1_batch, sent2_batch, sent1_lengths, sent2_lengths, _, _, _, target_classes = next(
                    train_generator)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                train_generator = iter(train_loader)
                sent1_batch, sent2_batch, sent1_lengths, sent2_lengths, _, _, _, target_classes = next(
                    train_generator)

            # calculate classifiy predictions (default classification=False (regression predictions))
            predictions, attention_matrix = model.forward(sent1_batch, sent2_batch, sent1_lengths, sent2_lengths,
                                                          classification=classify)

            target_classes = torch.tensor(target_classes)
            target_classes = Variable(
                target_classes.float(), requires_grad=True)

            batch_train_loss = loss(predictions, target_classes.long()) + attention_penalty_loss(attention_matrix,
                                                                                                 self_attention_config['penalty'], device)

            optimizer.zero_grad()  # or  model.zero_grad(set_to_none=True)
            batch_train_loss.backward()

            # gradient clipping before optimizer step
            if clip:

                torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)

            optimizer.step()
            total_loss += batch_train_loss

            correct += torch.eq(torch.max(predictions, 1)
                                [1], target_classes).data.sum()
            y_true += list(target_classes)
            y_pred += list(torch.max(predictions, 1)[1])

        scheduler.step()

        # computing accuracy of classification

        acc = correct/(batch_train_num*train_loader.batch_size)

        # print results every 100 epochs
        if True:  # epoch % 5 == 0:
            print('[%d/%d] train_loss: %.3f, train_accuracy: %.3f ' %
                  (epoch, max_epochs - 1, total_loss.data.float()/batch_train_num, acc))
        if epoch == max_epochs - 1:
            print('Final accuracy: %.3f, expected %.3f ' % (acc, 1.0))

    return model


# Target task training functions

def target_train_model(model, optimizer, scheduler, dataloader, max_epochs, config_dict, clip=False):
    device = config_dict["device"]
    self_attention_config = config_dict['self_attention_config']
    mse = nn.MSELoss()
    max_score = -1
    train_loader, _, _ = dataloader
    train_generator = iter(train_loader)
    batch_train_num = len(train_loader)

    # save information for visualization
    dictionary_info = {}
    dictionary_info['train_loss'] = []
    dictionary_info['val_loss'] = []
    dictionary_info['train_score'] = []
    dictionary_info['val_score'] = []
    dictionary_info['epochs'] = []

    for epoch in tqdm(range(max_epochs)):
        y_true = list()
        y_pred = list()
        total_loss = 0

        # iterate over train batches
        for batch in range(batch_train_num):

            # protection block for restarting sampling in case dataloader stops
            try:
                # batch Samples
                sent1_batch, sent2_batch, sent1_lengths, sent2_lengths, targets, _, _, _ = next(
                    train_generator)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                train_generator = iter(train_loader)
                sent1_batch, sent2_batch, sent1_lengths, sent2_lengths, targets, _, _, _ = next(
                    train_generator)

            # calculate predictions and attended matrix for penalizing term
            predictions, attention_matrix = model.forward(
                sent1_batch, sent2_batch, sent1_lengths, sent2_lengths)

            # Protect loss calculations (mse loss + penalty term)
            try:

                targets = Variable(targets, requires_grad=True)
                batch_train_loss = mse(predictions.float(), targets.float()) + attention_penalty_loss(attention_matrix,
                                                                                                      self_attention_config['penalty'], device)

            except RuntimeError:

                raise Exception(
                    "nan values on regularization. Rremove regularization or add very small values")

            optimizer.zero_grad()  # or  model.zero_grad(set_to_none=True)
            batch_train_loss.backward()

            # gradient clipping before optimizer step
            if clip:

                torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)

            optimizer.step()
            total_loss += batch_train_loss

            y_true += list(targets.detach().numpy())
            y_pred += list(predictions.detach().numpy())

        scheduler.step()

        # computing accuracy using pearson correlation score, Best possible score is 1.0
        # and it can be negative (because the model can be arbitrarily worse)

        # explained_variance_score(y_true, y_pred)
#         r2score = r2_score(y_true, y_pred)
        score = np.corrcoef(y_true, y_pred)[0, 1]

        # compute model metrics on dev set
        val_score, val_loss, val_pvalue = evaluate_dev_set(
            model,  mse, dataloader, config_dict, device
        )

        if val_score > max_score:
            max_score = val_score
            logging.info(
                "new model saved")

            # save the model if it is better than the prior best
            torch.save(model.state_dict(), "{}.pth".format(
                config_dict["model_name"]))

        logging.info(
            "Train loss: {} - Train pearson score: {} -- Validation loss: {} - Validation pearson score: {}- Validation p_value: {}".format(
                total_loss.data.float()/batch_train_num, score, val_loss, val_score, val_pvalue
            )
        )

        # print results every 100 epochs
        if True:  # epoch % 5 == 0:
            print('[%d/%d] train_loss: %.3f, train_score: %.3f ' %
                  (epoch, max_epochs - 1, total_loss.data.float()/batch_train_num, score))
        if epoch == max_epochs - 1:
            print('Final score: %.3f, expected %.3f' %
                  (score, 1.0))

        # save progress in a dictionary for ploting purposes

        dictionary_info['train_loss'].append(
            total_loss.data.float().item()/batch_train_num)
        dictionary_info['val_loss'].append(val_loss.item())
        dictionary_info['epochs'].append(epoch)
        dictionary_info['train_score'].append(score)
        dictionary_info['val_score'].append(val_score)

        # save results to txt files
        if epoch == max_epochs - 1:
            now = datetime.now().time()  # time object
            now = str(now)
            textfile = open("logs/"+now+"_losses&scores.txt", "w")
            textfile.write(" epochs : " + str(dictionary_info['epochs']))
            textfile.write("\n\n train_losses : " +
                           str(dictionary_info['train_loss']))
            textfile.write("\n\n val_losses : " +
                           str(dictionary_info['val_loss']))
            textfile.write("\n\n train_scores : " +
                           str(dictionary_info['train_score']))
            textfile.write("\n\n val_scores : " +
                           str(dictionary_info['val_score']))
            textfile.close()

    # Visualize  progress
    plot_progress(dictionary_info, max_epochs)
    return model


# Eevaluate model on val data split for hyperparameter tuning
def evaluate_dev_set(model, criterion, data_loader, config_dict, device):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on dev set")

    device = config_dict["device"]
    self_attention_config = config_dict['self_attention_config']
    _, val_loader, _ = data_loader
    val_generator = iter(val_loader)
    batch_val_num = len(val_loader)
    y_true = list()
    y_pred = list()
    total_loss = 0

    # iterate over train batches
    for batch in range(batch_val_num):

        # protection block for restarting sampling in case dataloader stops
        try:
            # Samples the batch
            sent1_batch, sent2_batch, sent1_lengths, sent2_lengths, targets, _, _, _ = next(
                val_generator)

        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            val_generator = iter(val_loader)
            sent1_batch, sent2_batch, sent1_lengths, sent2_lengths, targets, _, _, _ = next(
                val_generator)

        predictions, attention_matrix = model.forward(
            sent1_batch, sent2_batch, sent1_lengths, sent2_lengths)
        targets = Variable(targets, requires_grad=True)

        try:

            # evaluate penalized loss (loss + penalty term)
            batch_val_loss = criterion(predictions.float(), targets.float()) + (attention_penalty_loss(attention_matrix,
                                                                                                       self_attention_config['penalty'], device))
        except RuntimeError:

            raise Exception(
                "nan values on regularization. Rremove regularization or add very small values")

        # save avg loss for batches into list
        total_loss += batch_val_loss
        y_true += list(targets.detach().numpy())
        y_pred += list(predictions.detach().numpy())

    _, p_value = spearmanr(y_true, y_pred, nan_policy='omit')
    val_score = np.corrcoef(y_true, y_pred)[0, 1]
    val_loss = (total_loss.data.float()/batch_val_num)
    return val_score, val_loss, p_value


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

    batch_size, attention_out_size = annotation_weight_matrix.size(
        0), annotation_weight_matrix.size(1)
    annotation_weight_matrix_trans = annotation_weight_matrix.transpose(1, 2)
    identity = torch.eye(attention_out_size)
    identity = Variable(identity.unsqueeze(0).expand(
        batch_size, attention_out_size, attention_out_size))
    annotation_mul_difference = annotation_weight_matrix@annotation_weight_matrix_trans - identity
    penalty = frobenius_norm(annotation_mul_difference)
    regulizing_term = penalty_coef*penalty

    return regulizing_term


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
    return torch.sqrt(torch.sum(annotation_mul_difference**2))
