from itertools import cycle
from time import time

import matplotlib.pyplot as plt
import pandas as pd
from scipy import interp
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from beat_the_benchmark_classifier import BeatTheBenchmarkClassifier
from data import Data
from w2v import *

# Resource related properties -------------------------------------------------
RESOURCES_PATH = "../resources/"
# TRAIN_DATASET = RESOURCES_PATH + "1k_train_set.csv"
TRAIN_DATASET = RESOURCES_PATH + "train_set.csv"
TEST_DATASET = RESOURCES_PATH + "test_set.csv"
DOC_TITLE_IDX = 2
DOC_CONTENT_IDX = 3
OUTPUT_CSV = "./EvaluationMetric_10fold.csv"
OUTPUT_PNG = "./roc_10fold.png"


def classification_test(model, pipeline_name, allData, _X_train=None):
    if _X_train is None:
        X_train = allData.X_train
    else:
        X_train = _X_train

    y_train = allData.y_train
    y_train_bin = allData.y_train_bin

    print("\tRunning 10-Fold test for: " + str(pipeline_name))
    k_fold = StratifiedKFold(n_splits=10, shuffle=False, random_state=20170218)

    scores = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro']

    metrics = {
        score: cross_val_score(model, X_train, y_train, cv=k_fold, n_jobs=-1, scoring=score).mean()
        for score in scores}

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for train_index, test_index in k_fold.split(X_train, y_train):
        _X_train, _X_test = X_train[train_index], X_train[test_index]
        _y_train, _y_test = y_train_bin[train_index], y_train_bin[test_index]

        clf = OneVsRestClassifier(model)
        clf.fit(_X_train, _y_train)
        if hasattr(model, "predict_proba"):
            probas_ = clf.predict_proba(_X_test)
        else:
            probas_ = clf.decision_function(_X_test)
            probas_ = (probas_ - probas_.min()) / (probas_.max() - probas_.min())

        fpr, tpr, _ = roc_curve(_y_test.ravel(), probas_.ravel())
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

    mean_tpr /= k_fold.get_n_splits()
    mean_tpr[-1] = 1.0

    metrics['roc_tpr_micro'] = mean_tpr
    metrics['roc_fpr_micro'] = mean_fpr
    metrics['roc_auc_micro'] = auc(mean_fpr, mean_tpr)

    return metrics


def plot_roc_curve(metrics):
    colors = cycle(['aqua', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2

    plt.figure()

    for class_name, color in zip(metrics, colors):
        metric = metrics[class_name]
        plt.plot(metric['roc_fpr_micro'], metric['roc_tpr_micro'], color=color, lw=lw,
                 label='{0} (AUC = {1:0.2f})'
                       ''.format(class_name, metric['roc_auc_micro']))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves of Different Classifiers')
    plt.legend(loc="lower right")

    plt.savefig(OUTPUT_PNG, bbox='tight')


if __name__ == '__main__':
    total_time_start = time()
    allData = Data(TRAIN_DATASET, TEST_DATASET)

    n_features = 200

    classifier_feat_extraction_name = []
    classifier_method_idx = 0
    metrics_scores = {}

    # SVM (BoW)
    classifier_feat_extraction_name.append('SVM (BoW)')
    pipeline = Pipeline([
        ('bow', CountVectorizer(max_features=n_features)),
        ('clf', LinearSVC(C=1.0))
    ])
    t1 = time()
    metrics_scores[classifier_feat_extraction_name[classifier_method_idx]] = \
        classification_test(pipeline, classifier_feat_extraction_name[classifier_method_idx], allData)
    t2 = time()
    print(classifier_feat_extraction_name[classifier_method_idx] + " = " + str(round(t2 - t1)) + "s")
    classifier_method_idx += 1
    print("________________________________________")

    # RandomForest (BoW)
    classifier_feat_extraction_name.append('RandomForest (BoW)')
    pipeline = Pipeline([
        ('bow', CountVectorizer(max_features=n_features)),
        ('forest', RandomForestClassifier(n_estimators=20))
    ])
    t1 = time()
    metrics_scores[classifier_feat_extraction_name[classifier_method_idx]] = \
        classification_test(pipeline, classifier_feat_extraction_name[classifier_method_idx], allData)
    t2 = time()
    print(classifier_feat_extraction_name[classifier_method_idx] + " = " + str(round(t2 - t1)) + "s")
    classifier_method_idx += 1
    print("________________________________________")

    # SVD
    # for 90% of features
    n_samples, n_components = CountVectorizer(max_features=n_features).fit_transform(allData.X_train,
                                                                                     allData.y_train).shape
    svd_algorithm = 'arpack'
    if svd_algorithm == 'arpack':
        n_components = min(int(n_components * 0.9), int(n_samples / 2))
    else:
        n_components = int(n_components * 0.9)

    # SVM (SVD)
    # algorithm = 'randomized' (default) crashes scikit https://github.com/scikit-learn/scikit-learn/issues/8183
    classifier_feat_extraction_name.append('SVM (SVD)')
    pipeline = Pipeline([
        ('bow', CountVectorizer(max_features=n_features)),
        # ('bow', CountVectorizer(ngram_range=(1, 2), max_features=200)),
        ('lsa', TruncatedSVD(n_components=n_components, algorithm=svd_algorithm)),
        ('clf', LinearSVC(C=1.0))
    ])
    t1 = time()
    metrics_scores[classifier_feat_extraction_name[classifier_method_idx]] = \
        classification_test(pipeline, classifier_feat_extraction_name[classifier_method_idx], allData)
    t2 = time()
    print(classifier_feat_extraction_name[classifier_method_idx] + " = " + str(round(t2 - t1)) + "s")
    classifier_method_idx += 1
    print("________________________________________")

    # RandomForest (SVD)
    classifier_feat_extraction_name.append('RandomForest (SVD)')
    pipeline = Pipeline([
        ('bow', CountVectorizer(max_features=n_features)),
        # ('bow', CountVectorizer(ngram_range=(1, 2), max_features=200)),
        ('lsa', TruncatedSVD(n_components=n_components, algorithm=svd_algorithm)),
        ('clf', RandomForestClassifier(n_estimators=20))
    ])
    t1 = time()
    metrics_scores[classifier_feat_extraction_name[classifier_method_idx]] = \
        classification_test(pipeline, classifier_feat_extraction_name[classifier_method_idx], allData)
    t2 = time()
    print(classifier_feat_extraction_name[classifier_method_idx] + " = " + str(round(t2 - t1)) + "s")
    classifier_method_idx += 1
    print("________________________________________")

    # SVM (W2V)
    classifier_feat_extraction_name.append('SVM (W2V)')
    pipeline = Pipeline([
        ('clf', LinearSVC(C=1.0))
    ])
    t1 = time()
    metrics_scores[classifier_feat_extraction_name[classifier_method_idx]] = \
        classification_test(pipeline, classifier_feat_extraction_name[classifier_method_idx], allData,
                            allData.X_train_w2v)
    t2 = time()
    print(classifier_feat_extraction_name[classifier_method_idx] + " = " + str(
        round(allData.w2v_duration + t2 - t1)) + "s")
    classifier_method_idx += 1
    print("________________________________________")

    # RandomForest (W2V)
    classifier_feat_extraction_name.append('RandomForest (W2V)')
    pipeline = Pipeline([
        ('clf', RandomForestClassifier(n_estimators=20))
    ])
    t1 = time()
    metrics_scores[classifier_feat_extraction_name[classifier_method_idx]] = \
        classification_test(pipeline, classifier_feat_extraction_name[classifier_method_idx], allData,
                            allData.X_train_w2v)
    t2 = time()
    print(classifier_feat_extraction_name[classifier_method_idx] + " = " + str(
        round(allData.w2v_duration + t2 - t1)) + "s")
    classifier_method_idx += 1
    print("________________________________________")

    # BeatTheBenchmarkClassifier (BTBC)
    classifier_feat_extraction_name.append('BTBC')
    docs_classification = BeatTheBenchmarkClassifier(data=allData)
    pipeline = docs_classification.classify()
    t1 = time()
    metrics_scores[classifier_feat_extraction_name[classifier_method_idx]] = \
        classification_test(pipeline, classifier_feat_extraction_name[classifier_method_idx], allData)
    t2 = time()
    print(classifier_feat_extraction_name[classifier_method_idx] + " = " + str(
        round(allData.w2v_duration + t2 - t1)) + "s")
    print("________________________________________")

    print("Plotting ROC Curve...")
    t1 = time()
    plot_roc_curve(metrics_scores)
    t2 = time()
    print("Completed plot in " + str(
        round(allData.w2v_duration + t2 - t1)) + "s")
    print("________________________________________")

    # Create pandas DataFrame to generate EvaluationMetric 10-fold CSV output.
    proper_labels = ['Accuracy', 'Precision', 'Recall', 'F-Measure', 'AUC']


    def rename_labels(metrics_dict, label, proper_label):
        metrics_dict[proper_label] = metrics_dict[label]
        del metrics_dict[label]


    for k, v in metrics_scores.items():
        del metrics_scores[k]['roc_fpr_micro']
        del metrics_scores[k]['roc_tpr_micro']
        # Rename metrics to proper ones
        rename_labels(metrics_scores[k], 'accuracy', proper_labels[0])
        rename_labels(metrics_scores[k], 'precision_micro', proper_labels[1])
        rename_labels(metrics_scores[k], 'recall_micro', proper_labels[2])
        rename_labels(metrics_scores[k], 'f1_micro', proper_labels[3])
        rename_labels(metrics_scores[k], 'roc_auc_micro', proper_labels[4])

    df = pd.DataFrame(metrics_scores, index=proper_labels, columns=classifier_feat_extraction_name)
    # Generate CSV output
    df.to_csv(OUTPUT_CSV, float_format="%.2f")

    total_time_end = time()
    print("Total time: " + str(round(total_time_end - total_time_start)) + "s")
    print("Done")
