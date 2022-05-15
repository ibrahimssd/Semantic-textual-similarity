from torch.utils.data import Dataset
from datasets import load_dataset, Dataset
import torch


"""
Standard Pytorch Dataset class for loading datasets.
"""


class STSDataset(Dataset):
    def __init__(
        self,
        sent1_tensor,
        sent2_tensor,
        target_tensor,
        sents1_length_tensor,
        sents2_length_tensor,
        raw_sents_1,
        raw_sents_2,
        sen_class,
    ):
        """
        initializes  and populates the the length, data and target tensors, and raw texts list
        """

        # We have made minor modifications in the assert block because targets and sentences have been passed as lists
        assert (


            len(sent1_tensor)
            == torch.tensor(list(target_tensor)).size(0)
            == len(sent2_tensor)
            == torch.tensor(list(sents1_length_tensor)).size(0)
            == torch.tensor(list(sents2_length_tensor)).size(0)
            == len(sen_class)
        )

        self.sent1_tensor = sent1_tensor
        self.sent2_tensor = sent2_tensor
        self.target_tensor = target_tensor
        self.sents1_length_tensor = sents1_length_tensor
        self.sents2_length_tensor = sents2_length_tensor
        self.raw_sents_1 = raw_sents_1
        self.raw_sents_2 = raw_sents_2
        self.sen_class = sen_class

    def __getitem__(self, index):
        """
        returns the tuple of data tensor, targets, lengths of sequences tensor and raw texts list
        """
        return (
            self.sent1_tensor[index],
            self.sent2_tensor[index],
            self.sents1_length_tensor[index],
            self.sents2_length_tensor[index],
            self.target_tensor[index],
            self.raw_sents_1[index],
            self.raw_sents_2[index],
            self.sen_class[index],
        )

    def __len__(self):
        """
        returns the length of the data tensor.
        """
        return torch.tensor(list(self.target_tensor)).size(0)
