import pandas as pd
from preprocess import Preprocess
import logging
import torch
from dataset import STSDataset
from datasets import load_dataset
from torchtext.legacy.data import Field
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)

"""
For loading STS data loading and preprocessing
"""


class STSData:
    def __init__(
        self,
        dataset_name,
        columns_mapping,
        stopwords_path="stopwords-en.txt",
        model_name="lstm",
        max_sequence_len=30,  # 512
        normalization_const=5.0,
        normalize_labels=True,
    ):
        """
        Loads data into memory and create vocabulary from text field.
        """
        self.normalize_labels = normalize_labels
        self.normalization_const = normalization_const
        self.normalize_labels = normalize_labels
        self.model_name = model_name
        self.max_sequence_len = max_sequence_len
        self.dataset_name = dataset_name
        # load data file into memory
        self.load_data(dataset_name, columns_mapping, stopwords_path)
        self.columns_mapping = columns_mapping
        # create vocabulary over entire dataset before train/test split
        self.create_vocab()

    def load_data(self, dataset_name, columns_mapping, stopwords_path):
        """
        Reads data set file from disk to memory using pandas
        """
        logging.info("loading and preprocessing data...")

        # load datasets
        sick_dataset = load_dataset(
            dataset_name, download_mode='reuse_cache_if_exists')

        # remove unneccessary rows
        sick_dataset = sick_dataset.remove_columns(['label', 'id', 'entailment_AB', 'entailment_BA', 'sentence_A_original',
                                                   'sentence_B_original', 'sentence_A_dataset', 'sentence_B_dataset'])
        train_pd = pd.DataFrame.from_dict(sick_dataset['train'])
        validation_pd = pd.DataFrame.from_dict(sick_dataset['validation'])
        test_pd = pd.DataFrame.from_dict(sick_dataset['test'])
        self.sick_dataframes = [train_pd, validation_pd, test_pd]
        # perform text preprocessing
        process = Preprocess(stopwords_path)
        self.processed_data = process.perform_preprocessing(
            self.sick_dataframes, columns_mapping)

        logging.info("reading and preprocessing data completed...")

    def create_vocab(self):
        """
        Creates vocabulary over entire text data field.
        """

        logging.info("creating vocabulary...")
        # create data frame for training dataset at index 0 in the dataset dictionary(self.processed_data)
        splits = list(self.processed_data.keys())
        training_dataset = self.processed_data[splits[0]]
        train_data = pd.DataFrame(training_dataset)

        # combine pairs of sentence in train data frame to build vocabulary
        cols = list(self.columns_mapping.values())
        cols.pop()
        train_data['sentenceA&B'] = train_data[cols].apply(
            lambda row: ' '.join(row.values.astype(str)), axis=1)

        data_field = Field(
            lower=True,
            include_lengths=True,
            pad_token='PAD',
            pad_first='SOS'
        )

        preprocessed_data = train_data['sentenceA&B'].apply(
            lambda x: data_field.preprocess(x))

        data_field.build_vocab(
            preprocessed_data,
            vectors='fasttext.simple.300d')

        # get the vocab instance
        self.vocab = data_field.vocab
        logging.info("creating vocabulary completed...")

    def data2tensors(self, data):
        """
        Converts raw data sequences into vectorized sequences as tensors
        """

        # convert sentences in data frames to tensors and add tensors to new column
        data['sent1_tensor'] = data['sent1_tensor'].apply(
            lambda lis: torch.as_tensor(lis))
        data['sent2_tensor'] = data['sent2_tensor'].apply(
            lambda lis: torch.as_tensor(lis))

        return data

    def get_data_loader(self, batch_size=8):

        train_data_df = pd.DataFrame(self.processed_data['train'])
        val_data_df = pd.DataFrame(self.processed_data['validation'])
        test_data_df = pd.DataFrame(self.processed_data['test'])

        # vectorization step
        train_data_df['sent1_tensor'] = train_data_df['sentence_A'].apply(
            lambda sen: self.vectorize_sequence(sen))
        train_data_df['sent2_tensor'] = train_data_df['sentence_B'].apply(
            lambda sen: self.vectorize_sequence(sen))

        val_data_df['sent1_tensor'] = val_data_df['sentence_A'].apply(
            lambda sen: self.vectorize_sequence(sen))
        val_data_df['sent2_tensor'] = val_data_df['sentence_B'].apply(
            lambda sen: self.vectorize_sequence(sen))

        test_data_df['sent1_tensor'] = test_data_df['sentence_A'].apply(
            lambda sen: self.vectorize_sequence(sen))
        test_data_df['sent2_tensor'] = test_data_df['sentence_B'].apply(
            lambda sen: self.vectorize_sequence(sen))

        # add tensor lengths column as a new column in data_frames
        train_data_df['sents1_length_tensor'] = train_data_df['sent1_tensor'].apply(
            lambda tensor: len(tensor))
        train_data_df['sents2_length_tensor'] = train_data_df['sent2_tensor'].apply(
            lambda tensor: len(tensor))

        val_data_df['sents1_length_tensor'] = val_data_df['sent1_tensor'].apply(
            lambda tensor: len(tensor))
        val_data_df['sents2_length_tensor'] = val_data_df['sent2_tensor'].apply(
            lambda tensor: len(tensor))

        test_data_df['sents1_length_tensor'] = test_data_df['sent1_tensor'].apply(
            lambda tensor: len(tensor))
        test_data_df['sents2_length_tensor'] = test_data_df['sent2_tensor'].apply(
            lambda tensor: len(tensor))

        # Normalize labels
        if (self.normalize_labels):

            train_data_df['relatedness_score'] = (train_data_df['relatedness_score'] - train_data_df['relatedness_score'].min())/(
                train_data_df['relatedness_score'].max() - train_data_df['relatedness_score'].min())

            val_data_df['relatedness_score'] = (val_data_df['relatedness_score'] - val_data_df['relatedness_score'].min())/(
                val_data_df['relatedness_score'].max() - val_data_df['relatedness_score'].min())

            test_data_df['relatedness_score'] = (test_data_df['relatedness_score'] - test_data_df['relatedness_score'].min())/(
                test_data_df['relatedness_score'].max() - test_data_df['relatedness_score'].min())

        # convert sentenses to PyTorch tensors to feed in the model
        train_data_df = self.data2tensors(train_data_df)
        val_data_df = self.data2tensors(val_data_df)
        test_data_df = self.data2tensors(test_data_df)

        # pad every sentence to max length so that all of them have the same length
        train_data_df['sent1_tensor'] = self.pad_sequences(
            train_data_df['sent1_tensor'])
        train_data_df['sent2_tensor'] = self.pad_sequences(
            train_data_df['sent2_tensor'])

        val_data_df['sent1_tensor'] = self.pad_sequences(
            val_data_df['sent1_tensor'])
        val_data_df['sent2_tensor'] = self.pad_sequences(
            val_data_df['sent2_tensor'])

        test_data_df['sent1_tensor'] = self.pad_sequences(
            test_data_df['sent1_tensor'])
        test_data_df['sent2_tensor'] = self.pad_sequences(
            test_data_df['sent2_tensor'])

        # creat data set object for each split
        train_dataset = STSDataset(train_data_df['sent1_tensor'],
                                   train_data_df['sent2_tensor'],
                                   train_data_df['relatedness_score'],
                                   train_data_df['sents1_length_tensor'],
                                   train_data_df['sents2_length_tensor'],
                                   train_data_df['sentence_A'],
                                   train_data_df['sentence_B']
                                   )

        val_dataset = STSDataset(val_data_df['sent1_tensor'],
                                 val_data_df['sent2_tensor'],
                                 val_data_df['relatedness_score'],
                                 val_data_df['sents1_length_tensor'],
                                 val_data_df['sents2_length_tensor'],
                                 val_data_df['sentence_A'],
                                 val_data_df['sentence_B']
                                 )

        test_dataset = STSDataset(test_data_df['sent1_tensor'],
                                  test_data_df['sent2_tensor'],
                                  test_data_df['relatedness_score'],
                                  test_data_df['sents1_length_tensor'],
                                  test_data_df['sents2_length_tensor'],
                                  test_data_df['sentence_A'],
                                  test_data_df['sentence_B']
                                  )

        # build data loaders for the three splits
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        return train_loader, val_loader, test_loader

    def sort_batch(self, batch, targets, lengths):
        """
        Sorts the data, lengths and target tensors based on the lengths
        of the sequences from longest to shortest in batch
        """
        sents1_lengths, perm_idx = lengths.sort(0, descending=True)
        sequence_tensor = batch[perm_idx]
        target_tensor = targets[perm_idx]
        return sequence_tensor.transpose(0, 1), target_tensor, sents1_lengths

    def vectorize_sequence(self, sentence):
        """
        Replaces tokens with their indices in vocabulary
        """
        splited_sentence = sentence.split()
        codes = [self.vocab[token] for token in splited_sentence]

        return codes

    def pad_sequences(self, sequences):
        """
         Pads zeros at the end of each sequence in data tensor till max
         length of sequence in that batch
         """
        # pad every split to the max length in the split
        num = len(sequences)
        max_len = self.max_sequence_len
        out_dims = (num, max_len, *sequences[0].shape[1:])
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        mask = sequences[0].data.new(*out_dims).fill_(0)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = 1
        return list(out_tensor)
