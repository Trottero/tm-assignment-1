# categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# %%
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import numpy as np
import pickle
import os
import csv
import pandas as pd


# %%

# Create dirs to organise results / models
os.makedirs('results', exist_ok=True)
os.makedirs('models',  exist_ok=True)

# Load in training dataset
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)

print('Number of classes: ', len(twenty_train.target_names))
print('Number of samples: ', len(twenty_train.data))

# %%


# Display a very basic sample
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])

# %%


# Model related functions
def get_pipeline(classifier, feature):
    text_clf = Pipeline([
        ('vect', feature),
        ('clf', classifier),
    ])

    return text_clf


def pickle_pipeline(pipeline, name):
    with open(f'models/{name}.pickle', 'wb') as f:
        pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_pipeline(name):
    with open(f'models/{name}.pickle', 'rb') as f:
        return pickle.load(f)


# %%

# Available modes + features
# We can still do a ton of tuning for these classifiers
classifiers = {
    'nbayes': MultinomialNB(),
    'svn': SGDClassifier(loss='hinge', penalty='l2',
                         alpha=1e-3, random_state=42,
                         max_iter=5, tol=None),
    'randomforest': RandomForestClassifier(max_depth=2, random_state=0)
}

features = {
    'count': CountVectorizer(),
    'tf': TfidfVectorizer(use_idf=False),
    'tf-idf': TfidfVectorizer()
}

# %%

# Fit classifiers for every configuration
print('Training models')
pbar = tqdm(total=len(classifiers) * len(features))
for classifiername, classifier in classifiers.items():
    for featurename, feature in features.items():
        pipeline = get_pipeline(classifier, feature)

        pipeline.fit(twenty_train.data, twenty_train.target)
        pickle_pipeline(pipeline, classifiername + featurename)
        pbar.update()

pbar.close()

# %%


# Test the models on a given test set.
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
docs_test = twenty_test.data

# Would probably be prettier if we were to just walk the models folder and use those values instead
for classifiername, classifier in classifiers.items():
    for featurename, feature in features.items():
        pipeline = unpickle_pipeline(classifiername + featurename)
        predicted = pipeline.predict(docs_test)
        # print out accuracy
        print(f'{classifiername}{featurename}', np.mean(predicted == twenty_test.target))
        report = metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names, output_dict=True)
        # This filters out the accuracy line which only has a single value
        report = {name: value.values() for name, value in report.items() if name != 'accuracy'}
        # Convert to dataframe for easy csv transformation
        df = pd.DataFrame.from_dict(report, orient='index', columns=['precision', 'recall', 'f1-score', 'support'])
        with open(f'results/{classifiername}{featurename}.csv', 'w', newline='') as f:
            f.writelines(df.to_csv())

# %%
