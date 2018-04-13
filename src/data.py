from time import time

import nltk.data
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import label_binarize

from w2v import *


class Data:
    def __init__(self, train_set_file, test_set_file):
        self.train_data = pd.read_csv(train_set_file, index_col='Id', sep='\t', encoding='utf-8')
        self.train_data['Content'] = self.train_data['Title'] + self.train_data['Content']
        self.test_data = pd.read_csv(test_set_file, index_col='Id', sep='\t', encoding='utf-8')

        self.train_size = int(self.train_data.shape[0] * 2 / 3)
        self.test_size = int(self.train_data.shape[0] - self.train_size)

        cat_set = set(self.train_data['Category'].values)
        cat_name_to_id = dict(zip(cat_set, range(len(cat_set))))
        self.cat_id_to_name = {v: k for k, v in cat_name_to_id.items()}

        self.train_data['IntCategory'] = np.asarray([cat_name_to_id[cat] for cat in self.train_data['Category'].values])

        self.n_classes = len(cat_set)
        self.allCategories_int = list(range(self.n_classes))

        self.X_train = self.train_data['Content'].values
        self.y_train = self.train_data['IntCategory'].values
        self.y_train_bin = label_binarize(y=self.y_train, classes=self.allCategories_int)

        self.X_test = self.test_data['Content'].values

        print("Preparing word2vec...")
        t0 = time()
        self.X_train_w2v = w2v_prepare(self)
        self.w2v_duration = time() - t0
        print("word2vec preparation completed in "
              + str(round(self.w2v_duration)) + "s")


def w2v_prepare(allData):
    train_data = allData.train_data

    try:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    except LookupError:
        nltk.download('punkt')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        nltk.download('stopwords')

    sentences = []
    for content in train_data['Content']:
        sentences += review_to_sentences(content, tokenizer)

    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 8  # Number of threads to run in parallel
    context = 10
    downsampling = 1e-3

    model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                     window=context,
                     sample=downsampling)
    model.init_sims(replace=True)

    clean_train_reviews = []
    for review in train_data["Content"]:
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
    return getAvgFeatureVecs(clean_train_reviews, model, num_features)
