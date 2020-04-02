import nltk
import pandas as pd
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

root = '/Users/alinajam/Desktop/quora_duplicate_questions.tsv'
df = pd.read_csv(root, delimiter='\t',  engine='python')


