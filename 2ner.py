import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn_crfsuite


df = pd.read_csv('ner_my_razmech.csv')  #наш размеченный файл
# print(df[:10])
print(df.groupby('Tag').size().reset_index(name='counts'))
X = df.drop('Tag', axis=1)
print(X.head())
# print(X[:10])
print(X.columns)
v = DictVectorizer(sparse=False)
X = v.fit_transform(X.to_dict('records'))
X.shape
y = df.Tag.values
classes = np.unique(y)
classes = classes.tolist()
print(classes)
X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
X_train.shape, y_train.shape
# Perceptron
new_classes = classes.copy()
new_classes.pop()
new_classes

# Get sentences
class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                           s['POS'].values.tolist(),
                                                           s['Tag'].values.tolist())]
        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(df)
sent = getter.get_next()
print(sent)
sentences = getter.sentences


#
# Features extraction
# Next, we extract more features (word parts, simplified POS tags, lower/title/upper flags, features of nearby words) and convert them to sklear-crfsuite format - each sentence should be converted to a list of dicts.

def word2features(sent, i):
    word = str(sent[i][0])
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'originalWord': word,
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],

    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, postag, label in sent]
def sent2tokens(sent):
    return [token for token, postag, label in sent]

X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
#Обучаем CRF
crf.fit(X_train, y_train)

sdf = pd.read_csv('ner_test.csv')
getter1 = SentenceGetter(sdf)
sent1 = getter.get_next()
# print(sent1)
sentences1 = getter1.sentences

X1 = [sent2features(s) for s in sentences1]

y_pred = crf.predict(X1)

# print("_______y_pred")
# print(y_pred)
mas_Tag = []
mas_text = []
for item in y_pred:
    for item1 in item:
        mas_Tag.append(item1)
for item in X1:
    for item1 in item:
        mas_text.append(item1["word.originalWord()"])
sdf['Tag'] = pd.Series(mas_Tag)
sdf['Word'] = pd.Series(mas_text)
# sdf.to_csv("ner_my.csv")
sdf.to_csv("ner_solved.csv")

# row = 0
# names_sre=[]
# str1=""
# for item in mas_Tag:
#     if item=="B-sre":
#         str1=mas_text[row]
#     if item == "I-sre":
#         str1 = str1 + " " +mas_text[row]
#     else:
#         if len(str1)>0:
#             names_sre.append(str1)
#             str1=""
#     row += 1
#
# print(names_sre)
# with open('ner_names_sre.txt', 'w') as f:
#     for item in names_sre:
#         f.write("%s\n" % item)

