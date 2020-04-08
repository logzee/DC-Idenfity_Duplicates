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
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

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

fs_1 = ['len_q1', 'len_q2', 'diff_len', 'len_char_q1',
        'len_char_q2', 'len_word_q1', 'len_word_q2',
        'common_words']

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

fs_2 = ['fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio',
       'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
       'fuzz_token_set_ratio']

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

n_splits = 10
kf = KFold(n_splits=n_splits)
cv_accuracies = []
for train, test in kf.split(data):
    x_train_data = np.array(data[fs_1+fs_2])[train]
    y_train_data = np.array(data["is_duplicate"])[train]
    x_test_data = np.array(data[fs_1+fs_2])[test]
    y_test_data = np.array(data["is_duplicate"])[test]

    classifier = linear_model.LogisticRegression(C=0.1,
                                 solver='sag', max_iter=100)

    classifier = classifier.fit(x_train_data, y_train_data)
    tempAccuracy = np.sum(classifier.predict(x_test_data) == y_test_data) / len(y_test_data)
    cv_accuracies.append(tempAccuracy)
average = sum(cv_accuracies)/n_splits

print('Accuracies with ' + str(n_splits) + '-fold cross validation: ')
for cv_accuracy in cv_accuracies:
    print(cv_accuracy)
average = (average + log_res_accuracy) / 2

print('The Logistic Regression model identifies duplicate sentences with an average accuracy of = ', average, '%')

detailedReport = classification_report(y_val, lr_preds)

print("Below is a detailed report of the models performance")
print(detailedReport)

# KNN = neighbors.KNeighborsClassifier(10, weights='uniform')
# KNN.fit(x_train, y_train)

# KNN_preds = KNN.predict(x_val)
# KNN_accuracy = np.sum(KNN_preds == y_val) / len(y_val)
# print("Nearest Neighbors accuracy: %0.3f" % KNN_accuracy)

# NB = naive_bayes.GaussianNB()
# NB = NB.fit(x_train, y_train)

# NB_preds = NB.predict(x_val)
# NB_accuracy = np.sum(NB_preds == y_val) / len(y_val)
# print("Gaussian Naive Bayes accuracy: %0.3f" % NB_accuracy)
print("Hello, please input two sentences. This program will determin if they are duplicate sentences or not.")
s1 = input("Enter sentence one: ")
s2 = input("Enter sentence two: ")

inputData = [[s1, s2, 0]]
inputDF = pd.DataFrame(inputData, columns=['question1', 'question2', 'is_duplicate'])
print(inputDF)

print("You entered: '", s1, "' and '", s2, "' please wait while the system processes...")

# length based features
inputDF['len_q1'] = inputDF.question1.apply(lambda x: len(str(x)))
inputDF['len_q2'] = inputDF.question2.apply(lambda x: len(str(x)))
# difference in lengths of two questions
inputDF['diff_len'] = inputDF.len_q1 - inputDF.len_q2

# character length based features
inputDF['len_char_q1'] = inputDF.question1.apply(lambda x:
len(''.join(set(str(x).replace(' ', '')))))
inputDF['len_char_q2'] = inputDF.question2.apply(lambda x:
len(''.join(set(str(x).replace(' ', '')))))

# word length based features
inputDF['len_word_q1'] = inputDF.question1.apply(lambda x:
len(str(x).split()))
inputDF['len_word_q2'] = inputDF.question2.apply(lambda x:
len(str(x).split()))

# common words in the two questions
inputDF['common_words'] = inputDF.apply(lambda x:
len(set(str(x['question1'])
.lower().split())
.intersection(set(str(x['question2'])
.lower().split()))), axis=1)

fs_1 = ['len_q1', 'len_q2', 'diff_len', 'len_char_q1',
        'len_char_q2', 'len_word_q1', 'len_word_q2',
        'common_words']

#fuzzyfeatures
inputDF['fuzz_qratio'] = inputDF.apply(lambda x: fuzz.QRatio(
    str(x['question1']), str(x['question2'])), axis=1)
inputDF['fuzz_WRatio'] = inputDF.apply(lambda x: fuzz.WRatio(
str(x['question1']), str(x['question2'])), axis=1)

inputDF['fuzz_partial_ratio'] = inputDF.apply(lambda x:
fuzz.partial_ratio(str(x['question1']),
str(x['question2'])), axis=1)

inputDF['fuzz_partial_token_set_ratio'] = inputDF.apply(lambda x:
fuzz.partial_token_set_ratio(str(x['question1']),
str(x['question2'])), axis=1)

inputDF['fuzz_partial_token_sort_ratio'] = inputDF.apply(lambda x:
fuzz.partial_token_sort_ratio(str(x['question1']),
str(x['question2'])), axis=1)

inputDF['fuzz_token_set_ratio'] = inputDF.apply(lambda x:
fuzz.token_set_ratio(str(x['question1']),
str(x['question2'])), axis=1)

fs_2 = ['fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio',
       'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
       'fuzz_token_set_ratio']

scaler = preprocessing.StandardScaler()
inputStringFeatures = inputDF[fs_1 + fs_2]
inputStringFeatures = inputStringFeatures.replace([np.inf, -np.inf], np.nan).fillna(0).values

inputPrediction = logres.predict(inputStringFeatures)
print(inputPrediction)
if(inputPrediction == 1):
    print("The sentences you inputed are duplicates")
else:
    print("The sentences you inputed are not duplictes")

