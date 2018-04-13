import re
from pprint import pprint
from time import time

import nltk
import pandas as pd
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data import Data

# Resource related properties -------------------------------------------------
RESOURCES_PATH = "../resources/"
TRAIN_DATASET = RESOURCES_PATH + "1k_train_set.csv"
# TRAIN_DATASET = RESOURCES_PATH + "100_train_set.csv"
TEST_DATASET = RESOURCES_PATH + "test_set.csv"
TEST_SET_CATEGORIES_OUTPUT_CSV = "testSet_categories.csv"


class BeatTheBenchmarkClassifier:
    def __init__(self, train_set_file=None, test_set_file=None, data=None):
        # Fields ----------------------------------------------------
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['one', 'two', 'three', 'first', 'also', 'since'])
        self.stemmer = SnowballStemmer("english")

        if data is None and train_set_file is not None and test_set_file is not None:
            self.data = Data(train_set_file, test_set_file)
        else:
            self.data = data

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

    def find_best_params(self, text_clf):
        # TODO Test with samples from the dataset:
        # http://stackoverflow.com/questions/9299346/fastest-svm-implementation-usable-in-python
        # Split the dataset
        parameters = {
            'vect__min_df': (0.1, 0.15, 0.2, 0.3),
            'vect__max_df': (0.8, 0.85, 0.9),
            'vect__ngram_range': ((1, 1), (1, 2)),  # uni-grams or bi-grams

            # 'clf__alpha': (0.00001, 0.000001),
            # 'clf__penalty': ('l2', 'elasticnet'),
        }
        grid_search = GridSearchCV(text_clf, parameters, n_jobs=-1)
        print("Performing grid search...")
        print("pipeline:", [name for name, _ in text_clf.steps])
        print("parameters:")
        pprint(parameters)
        t0 = time()
        grid_search.fit(self.data.X_train, self.data.y_train)
        print("done in %0.3fs" % (time() - t0))
        print("")

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
        return best_parameters

    def classify(self):
        # Define pipeline for text feature extraction with an SVM classifier
        text_clf = Pipeline([
            ('vect', CountVectorizer(max_features=200)),
            ('tfidf', TfidfTransformer()),
            ('clf', LinearSVC(C=1.0))
        ])
        best_parameters = self.find_best_params(text_clf)

        # Create new pipeline with best parameters
        text_clf_best_params = Pipeline([
            ('vect', CountVectorizer(
                max_features=200,
                min_df=best_parameters['vect__min_df'],
                max_df=best_parameters['vect__max_df'],
                ngram_range=best_parameters['vect__ngram_range'],
                stop_words=self.stop_words, tokenizer=self.tokenize_and_stem)),
            ('tfidf', TfidfTransformer()),
            ('clf', LinearSVC(C=1.0))
        ])

        print("\nFitting our model to the training data...")
        start = time()
        text_clf_best_params.fit(self.data.X_train, self.data.y_train)
        print("Completed in: " + str(round(time() - start)) + "s")

        print("\nPredicting test data categories...")
        start = time()
        predicted_test_data_category_ids = text_clf_best_params.predict(self.data.X_test)
        predicted_test_data_categories = \
            [self.data.cat_id_to_name[cat_id] for cat_id in predicted_test_data_category_ids]
        print("Completed in: " + str(round(time() - start)) + "s")

        # Create two series
        print("Generating CSV output for test data...")
        start = time()
        test_document_ids_series = pd.Series(self.data.test_data.index.values, name='Test_Document_ID')
        predicted_document_categ_series = pd.Series(predicted_test_data_categories, name='Predicted_Category')

        document_id_category_name_test_data = \
            pd.concat([test_document_ids_series, predicted_document_categ_series], axis=1)
        document_id_category_name_test_data.to_csv(TEST_SET_CATEGORIES_OUTPUT_CSV, sep='\t', index=False)
        print("Completed in: " + str(round(time() - start)) + "s")

        return text_clf_best_params


if __name__ == '__main__':
    total_start_time = time()
    docs_classification = BeatTheBenchmarkClassifier(train_set_file=TRAIN_DATASET, test_set_file=TEST_DATASET)
    docs_classification.classify()
    print("\nBeat-the-benchmark completed in: " + str(round(time() - total_start_time)) + "s")
