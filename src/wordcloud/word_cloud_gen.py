import argparse
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import nltk
from nltk import sent_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from wordcloud import WordCloud

from src.utils.file_utils import read_documents, file_len

# Resource related properties -------------------------------------------------
RESOURCES_PATH = "resources/"
DOCUMENT_IDX = 0
# Font resources properties ---------------------------------------------------
FONTS_PATH = RESOURCES_PATH + "fonts/"
FONT_OPEN_SANS = FONTS_PATH + "OpenSans-Light.ttf"
# Word Cloud related properties -----------------------------------------------
WORD_CLOUD_WIDTH = 2000
WORD_CLOUD_HEIGHT = 1200
WORD_CLOUD_MAX_FONT_SIZE = 240
WORD_CLOUD_FONT_STEP = 4
WORD_CLOUD_OUTPUT_FORMAT = "png"
WORD_CLOUD_OUTPUT_NAME = "word_cloud" + "." + WORD_CLOUD_OUTPUT_FORMAT
WORD_CLOUD_OUTPUT_FILENAME = RESOURCES_PATH + WORD_CLOUD_OUTPUT_NAME
WORD_CLOUD_OUTPUT_DPI = 1200
WORD_CLOUD_BACKGROUND_COLOR = "#ffffff"


class Corpus:
    def __init__(self, dataset_file, k, has_header, stop_words):
        # Fields ----------------------------------------------------
        self.dataset_file = dataset_file
        self.most_common_words_number = k
        self.has_header = has_header
        self.stop_words = stop_words
        self.stop_words.update(['one', 'two', 'three', 'first', 'last', 'also', 'since', '\s', 'could', 'would', 'may',
                                'much', 'says'])
        self.top_categories_with_frequencies = {}
        self.lemmatizer = WordNetLemmatizer()
        self.generate_word_cloud(DOCUMENT_IDX)

    def tokenize_and_lemmatize(self, text):
        tokens = [word for sent in sent_tokenize(text) for word in wordpunct_tokenize(sent)
                  if word not in self.stop_words and word.isalpha()]
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def generate_word_cloud(self, document_idx):
        # Used for displaying progress
        documents_count = file_len(self.dataset_file) - 1
        current_document = 0

        word_frequency_counter = Counter()

        # Tokenize text removing stopwords and accepting only alphabetic tokens
        for row in read_documents(self.dataset_file, self.has_header):
            if len(row) <= document_idx:
                continue
            lowered_document = row[document_idx].lower()
            tokenized_document = self.tokenize_and_lemmatize(lowered_document)

            word_frequency_counter.update(tokenized_document)

            current_document += 1
            update_progress(current_document / float(documents_count))

        # Remove words that appear only once
        # print("\nRemoving imposter (duplicate) words...")
        # word_frequency_dict = {k: v for k, v in word_frequency_counter.most_common(300).items() if v > 1}

        print("\nGenerating word cloud...")
        word_cloud = WordCloud(background_color=WORD_CLOUD_BACKGROUND_COLOR,
                               font_step=WORD_CLOUD_FONT_STEP,
                               font_path=FONT_OPEN_SANS,
                               max_font_size=WORD_CLOUD_MAX_FONT_SIZE,
                               width=WORD_CLOUD_WIDTH, height=WORD_CLOUD_HEIGHT)

        word_cloud.generate_from_frequencies(dict(word_frequency_counter.most_common(self.most_common_words_number)))
        print("Rendering ultra crisp word cloud...")
        plt.imshow(word_cloud)
        plt.axis("off")
        plt.savefig(WORD_CLOUD_OUTPUT_FILENAME, format=WORD_CLOUD_OUTPUT_FORMAT,
                    bbox_inches='tight', pad_inches=0, aspect='normal', dpi=WORD_CLOUD_OUTPUT_DPI)
        print("Word cloud was generated and saved as a file at:\n\t" + os.path.abspath(WORD_CLOUD_OUTPUT_FILENAME))
        plt.show()


# Utilities
def update_progress(amount_done):
    sys.stdout.write(
        "\rTokenizing documents: [{0:50s}] {1:.1f}%".format('#' * int(amount_done * 50), amount_done * 100))


def main(argv):
    input_file = None
    k = -1

    parser = argparse.ArgumentParser(description='Generate word cloud from CSV.')
    parser.add_argument('-i', '--input-file', type=str, default='wordcloud/example_input.txt', help='input CSV file',
                        required=False)
    parser.add_argument('-k', type=int, default=500,
                        help='k most common words to generate the word cloud [default: 500]')

    args = parser.parse_args()
    input_file = args.input_file
    k = args.k

    print("Will generate WordCloud for [" + input_file + "] with [" + str(k) + "] most common words.")
    Corpus(input_file, k, True, set(stopwords.words('english')))
    print("Done")


if __name__ == '__main__':
    if sys.version_info < (3, 0):
        print("python-version > 3.0 isn't satisfied, exiting...")
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet')
    main(sys.argv)
