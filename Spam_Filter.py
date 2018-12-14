# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 02:44:20 2018

@author: Pranav
"""

#Import the modules
import pandas as pd
import nltk
import re
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

#Read the data into a data-frame
df=pd.read_table('SMSSpamCollection.txt',header=None)
#df.head()
#df.info()

#Classify into spam and ham labels 
y=df[0]
label_encoder=preprocessing.LabelEncoder()
y_enc=label_encoder.fit_transform(y)

#Raw text data
raw=df[1]

#Normalization
#Replace email addresses
raw1=raw.str.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr')

#Replace URL
raw1=raw1.str.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr')

#Replace Symbols
raw1=raw1.str.replace(r'£|\$', 'moneysymb')    

#Replace Phone Numbers
raw1=raw1.str.replace(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr')  

#Replace numbers
raw1=raw1.str.replace(r'\d+(\.\d+)?', 'numbr')

#Replace punctuations, whitespaces and whitespaces
raw1=raw1.str.replace(r'[^\w\d\s]', ' ')
raw1=raw1.str.replace(r'\s+', ' ')
raw1=raw1.str.replace(r'^\s+|\s+?$', '')
raw1=raw1.str.lower()

#Remove stop words
stop_words=nltk.corpus.stopwords.words('english')
raw1=raw1.apply(lambda x: ' '.join(term for term in x.split() if term not in set(stop_words)))

#Stemming
p=nltk.PorterStemmer()
raw1=raw1.apply(lambda x: ' '.join(p.stem(term) for term in x.split()))

#This function is to enable real time input to the classifier
#Function for cleaning/preprocessing the text data
#Normalization
#Removing Stop Words
#Stemming
def process(in_string):
    assert(type(in_string) == str)
    #Stemming and stop words
    p=nltk.PorterStemmer()
    stop=nltk.corpus.stopwords.words('english')
    
    #Replace Email Addresses
    str1=re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', in_string)
    
    #Replace URL
    str1=re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', str1)
    
    #Replace symbols
    str1=re.sub(r'£|\$', 'moneysymb', str1)
    
    #Replace Phone Numbers
    str1=re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', str1)
    
    #Replace numbers
    str1=re.sub(r'\d+(\.\d+)?', 'numbr', str1)
    
    #Replace punctuations and whitespaces
    str1=re.sub(r'[^\w\d\s]', ' ', str1)
    str1=re.sub(r'\s+', ' ', str1)
    str1=re.sub(r'^\s+|\s+?$', '', str1.lower())
    
    #Return the processed string
    return ' '.join(p.stem(term) for term in str1.split() if term not in set(stop))

#Tokenization using n-grams model for feature extraction
vect=TfidfVectorizer(ngram_range=(1,2))
X_ngrams=vect.fit_transform(raw1)
#X_ngrams.shape

#Split data into testing and training datasets
X_train, X_test, y_train, y_test = train_test_split(X_ngrams, y_enc, test_size=0.25, random_state=42, stratify=y_enc)

#Linear Support Vector Classifier 
clf = svm.LinearSVC(loss='hinge')
clf.fit(X_train, y_train)

#Predict
y_pred = clf.predict(X_test)
acc=metrics.f1_score(y_test, y_pred)
print('Accuracy of classifier is {}'.format(acc))
print('\n')
cm=metrics.confusion_matrix(y_test, y_pred)
print('Confusion Matrix is')
print(cm)
print('\n')

#Custom Confusion Matrix
pd.DataFrame(metrics.confusion_matrix(y_test, y_pred), 
             index=[['actual', 'actual'], ['spam', 'ham']], 
             columns=[['predicted', 'predicted'], ['spam', 'ham']])


#Function to sum it all up !!
#Execute this function with a real line of text data for real time predictions
def spam_filter(message):
    if clf.predict(vect.transform([process(message)])):
        return 'spam'
    else:
        return 'not spam'
    

example = """  Please call our customer service representative on FREEPHONE 
    0808 145 4742 between 9am-11pm as you have WON
    a guaranteed £1000 cash or £5000 prize!  """
print('Example string is \n')
print(example)
print('\n')
    
#Just invoke this function to see whether a given input string is spam or not
print('Prediction for the example is {}'.format(spam_filter(example)))















