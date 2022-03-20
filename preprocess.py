import re
import string
from datasets import load_dataset, Dataset
import pandas as pd

"""
Performs basic text cleansing on the unstructured field 
"""


class Preprocess:
    def __init__(self, stpwds_file_path):
        """
        Initializes regex patterns and loads stopwords
        """
        # Save stop words in a list and compile a regular expression pattern provided as a string into a regex pattern object
        with open(stpwds_file_path) as fh:
            self.stopwords = list(set(fh.read().split()))
        self.noise_re = re.compile('\\b(%s)\\W' % (
            '|'.join(map(re.escape, self.stopwords))), re.I)

    # preprocess and clean text data

    def perform_preprocessing(self, data, columns_mapping):
        sen_A = columns_mapping['sent1']
        sen_B = columns_mapping['sent2']
        score = columns_mapping['label']
        cleaned_data = []
        for data_frame in data:

            # extract two sentences groups in a list
            groupA = list(data_frame[sen_A])
            groupB = list(data_frame[sen_B])
            # normalize text to lower case
            groupA = [x.lower() for x in groupA]
            groupB = [x.lower() for x in groupB]
            # remove punctuations
            groupA = [''.join(c for c in x if c not in string.punctuation)
                      for x in groupA]
            groupB = [''.join(c for c in x if c not in string.punctuation)
                      for x in groupB]
            # remove stopwords
            groupA = [self.noise_re.sub('', p) for p in groupA]
            groupB = [self.noise_re.sub('', p) for p in groupB]
            # Trim extra whitespace
            groupA = [' '.join(x.split()) for x in groupA]
            groupB = [' '.join(x.split()) for x in groupB]
            # Remove numbers
            groupA = [''.join(c for c in x if c not in '0123456789')
                      for x in groupA]
            groupB = [''.join(c for c in x if c not in '0123456789')
                      for x in groupB]
            # return data_back to DataFrame
            data_frame[sen_A] = groupA
            data_frame[sen_B] = groupB
            cleaned_data.append(data_frame)

        # form dictionary for the three splits
        sick_dataset = {'train': Dataset.from_pandas(cleaned_data[0]),
                        'validation': Dataset.from_pandas(cleaned_data[1]),
                        'test': Dataset.from_pandas(cleaned_data[2])}

        # data frame for all splits combined and cleaned
        data_frame = pd.concat(cleaned_data, ignore_index=True)

        return sick_dataset
