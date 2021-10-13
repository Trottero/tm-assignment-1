# categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# %%
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import naive_bayes
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
twenty_train = fetch_20newsgroups(
    subset='train', shuffle=True, random_state=42)

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
        print(f'{classifiername}{featurename}',
              np.mean(predicted == twenty_test.target))
        report = metrics.classification_report(
            twenty_test.target, predicted, target_names=twenty_test.target_names, output_dict=True)
        # This filters out the accuracy line which only has a single value
        report = {name: value.values()
                  for name, value in report.items() if name != 'accuracy'}
        # Convert to dataframe for easy csv transformation
        df = pd.DataFrame.from_dict(report, orient='index', columns=[
                                    'precision', 'recall', 'f1-score', 'support'])
        with open(f'results/{classifiername}{featurename}.csv', 'w', newline='') as f:
            f.writelines(df.to_csv())

# %%
# merge all csv files into 1 file with latex table formatting

# iterate over all results files
result_dir = "results"
col_names = ["", "precision", "recall", "f1-score"]
themes = twenty_train.target_names

with open("all_results_latex_table_format.txt", "w") as complete_result_f:
    complete_result_f.write("\\hline \n")
    complete_result_f.write("&".join(col_names)+"\\\ \\hline \n")
    for filename in os.listdir(result_dir):
        # check to be sure we read the right files
        if filename.endswith(".csv"):
            with open(os.path.join(result_dir, filename), "r") as subresult_f:
                # only take results of precision, recall, f1 of macro avg
                # and round each num to 3 digits, and enclose within $$ so it will
                # be displayed nicely in table
                macro_avg_formatted = [f"${str(round(float(num), 3))}$" for num in subresult_f.readlines()[-2].strip().split(
                    ",")[1:-1]]
                complete_result_f.write(
                    "&".join([filename[:-4]]+macro_avg_formatted)+"\\\ \\hline \n")


# %%
# try out different countvectorizer features. for which one ?
# for unigrams: less features than texts in corpus

parameters = {
    'vectorizer__lowercase': (True, False),
    'vectorizer__stop_words': ('english', None),
    'vectorizer__analyzer': ('word', 'char', 'char_wb'),
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'vectorizer__max_features': (10, 20, 50, 100, 500, 1000, , 1500, None)
}

text_clf = Pipeline(
    [('vectorizer', TfidfVectorizer()), ('clf', MultinomialNB())])

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])


# %%
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


# %%
