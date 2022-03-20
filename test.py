from scipy.stats import spearmanr
import logging
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from train import attention_penalty_loss


def evaluate_test_set(model, data_loader, config_dict):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating score on test set")

    device = config_dict["device"]
    self_attention_config = config_dict['self_attention_config']
    _, _, test_loader = data_loader
    test_generator = iter(test_loader)
    num_test_batch = len(test_loader)
    y_true = list()
    y_pred = list()
    total_abs_loss = 0

    # iterate over test batches
    for batch in range(num_test_batch):

        # protect dataloader sampling process
        try:
            #  batch sampling
            sent1_batch, sent2_batch, sent1_lengths, sent2_lengths, targets, raw_sent1, raw_sent2 = next(
                test_generator)

        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            test_generator = iter(test_loader)
            sent1_batch, sent2_batch, sent1_lengths, sent2_lengths, targets, raw_sent1, raw_sent2 = next(
                test_generator)

        predictions, attention_matrix = model.forward(
            sent1_batch, sent2_batch, sent1_lengths, sent2_lengths)

        # protect loss calculations from nan values
        try:
            # targets= Variable(targets, requires_grad=True)
            abs_batch_test_loss = mean_absolute_error(targets.detach().numpy(), predictions.detach().numpy()) + (attention_penalty_loss(attention_matrix,
                                                                                                                                        self_attention_config['penalty'], device))

        except RuntimeError:

            raise Exception("nan values issue ")

        # aggregate loss for test batches
        total_abs_loss += abs_batch_test_loss

        y_true += list(targets.detach().numpy())
        y_pred += list(predictions.detach().numpy())

    spearman_score, _ = spearmanr(y_true, y_pred, nan_policy='omit')
    pearson_score = np.corrcoef(y_true, y_pred)[0, 1]

    # print absolute error , spearman and pearson correlation
    print('abs_batch_test_loss: %.3f, spearman_score: %.3f , pearson_score: %.3f' % (
        total_abs_loss/num_test_batch, spearman_score, pearson_score))
