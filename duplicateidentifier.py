import nltk
import pandas as pd
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

root = '/Users/alinajam/Desktop/quora_duplicate_questions.tsv'
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
       'fuzz_token_set_ratio', 'fuzz_token_sort_ratio']

