import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from utils.file_utils import read_documents, file_len

# import mpld3

# Resource related properties -------------------------------------------------
RESOURCES_PATH = "../resources/"
DATASET = RESOURCES_PATH + "train_set.csv"
DOC_TITLE_IDX = 2
DOC_CONTENT_IDX = 3


class Corpus:
    def __init__(self, dataset_file, has_header, stop_words):
        # Fields ----------------------------------------------------
        self.dataset_file = dataset_file
        self.has_header = has_header
        self.stop_words = stop_words
        self.stop_words.update(['one', 'two', 'three', 'first', 'also', 'since'])
        self.top_categories_with_frequencies = {}
        self.stemmer = SnowballStemmer("english")
        self.kmeans(DOC_TITLE_IDX, DOC_CONTENT_IDX)

    def kmeans(self, doc_title_idx, doc_cont_idx):
        # Used for displaying progress
        documents_count = file_len(self.dataset_file) - 1
        current_document = 0
        print(documents_count)

        totalvocab_stemmed = []
        totalvocab_tokenized = []
        for row in read_documents(self.dataset_file, self.has_header):
            content = row[doc_cont_idx]

            allwords_stemmed = self.tokenize_and_stem(content)
            totalvocab_stemmed.extend(allwords_stemmed)

            allwords_tokenized = self.tokenize_only(content)
            totalvocab_tokenized.extend(allwords_tokenized)

            vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)
            print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

    def tokenize_and_stem(self, text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [self.stemmer.stem(t) for t in filtered_tokens]
        return stems

    def tokenize_only(self, text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        return filtered_tokens


if __name__ == '__main__':
    corpus = Corpus(DATASET, True, set(stopwords.words('english')))
    print("Done")
