import nltk
import numpy as np
import pandas as pd

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import  preprocessing
from sklearn import neighbors, naive_bayes, svm, linear_model

root = '/Users/mcint/comp395/HW/DC4/quora_duplicate_questions.tsv'
data = pd.read_csv(root, delimiter='\t',  engine='python')

data = data.drop(['id', 'qid1', 'qid2'], axis=1)
# length based features
data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
# difference in lengths of two questions
data['diff_len'] = data.len_q1 - data.len_q2

# character length based features
data['len_char_q1'] = data.question1.apply(lambda x:
len(''.join(set(str(x).replace(' ', '')))))
data['len_char_q2'] = data.question2.apply(lambda x:
len(''.join(set(str(x).replace(' ', '')))))

# word length based features
data['len_word_q1'] = data.question1.apply(lambda x:
len(str(x).split()))
data['len_word_q2'] = data.question2.apply(lambda x:
len(str(x).split()))

# common words in the two questions
data['common_words'] = data.apply(lambda x:
len(set(str(x['question1'])
.lower().split())
.intersection(set(str(x['question2'])
.lower().split()))), axis=1)

fs_1 = data[['len_q1', 'len_q2', 'diff_len', 'len_char_q1',
        'len_char_q2', 'len_word_q1', 'len_word_q2',
        'common_words']]

print("Basic Features: ", fs_1)
#fuzzyfeatures
data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(
    str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(
str(x['question1']), str(x['question2'])), axis=1)

data['fuzz_partial_ratio'] = data.apply(lambda x:
fuzz.partial_ratio(str(x['question1']),
str(x['question2'])), axis=1)

data['fuzz_partial_token_set_ratio'] = data.apply(lambda x:
fuzz.partial_token_set_ratio(str(x['question1']),
str(x['question2'])), axis=1)

data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x:
fuzz.partial_token_sort_ratio(str(x['question1']),
str(x['question2'])), axis=1)

data['fuzz_token_set_ratio'] = data.apply(lambda x:
fuzz.token_set_ratio(str(x['question1']),
str(x['question2'])), axis=1)

fs_2 = data[['fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio',
       'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
       'fuzz_token_set_ratio']]

print("Fuzzy Features: ", fs_2)

#logisticregression

scaler = preprocessing.StandardScaler()
y = data.is_duplicate.values
y = y.astype('float32').reshape(-1, 1)
X = data[fs_1+fs_2]
X = X.replace([np.inf, -np.inf], np.nan).fillna(0).values
X = scaler.fit_transform(X)

np.random.seed(42)
n_all, _ = y.shape
idx = np.arange(n_all)
np.random.shuffle(idx)
n_split = n_all // 10
idx_val = idx[:n_split]
idx_train = idx[n_split:]
x_train = X[idx_train]
y_train = np.ravel(y[idx_train])
x_val = X[idx_val]
y_val = np.ravel(y[idx_val])

logres = linear_model.LogisticRegression(C=0.1,
                                 solver='sag', max_iter=1000)
logres.fit(x_train, y_train)
lr_preds = logres.predict(x_val)
log_res_accuracy = np.sum(lr_preds == y_val) / len(y_val)
print("Logistic regr accuracy: %0.3f" % log_res_accuracy)

KNN = neighbors.KNeighborsClassifier(10, weights='uniform')
KNN.fit(x_train, y_train)

KNN_preds = KNN.predict(x_val)
KNN_accuracy = np.sum(KNN_preds == y_val) / len(y_val)
print("Nearest Neighbors accuracy: %0.3f" % KNN_accuracy)

NB = naive_bayes.GaussianNB()
NB = NB.fit(x_train, y_train)

NB_preds = NB.predict(x_val)
NB_accuracy = np.sum(NB_preds == y_val) / len(y_val)
print("Gaussian Naive Bayes accuracy: %0.3f" % NB_accuracy)

