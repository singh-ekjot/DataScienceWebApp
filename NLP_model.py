#!/usr/bin/env python
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import re
import nltk #natural language toolkit
import sklearn


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer


# In[5]:


nltk.download('stopwords')


# In[6]:



from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[7]:


file='E:/Placement/Projects/Python projects/Data Science Web App/spam.csv'
#check encoding to avoid encoding error while reading file with pandas
#import chardet
#with open(file, 'rb') as rawdata:
 #   result = chardet.detect(rawdata.read(100000))
#result


# In[8]:


#NLP Model
df = pd.read_csv(file,encoding='Windows-1252')
#df.head()


# In[9]:


df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)


# In[10]:


#df.head()


# In[11]:


df.rename(columns = {'v1':'labels','v2':'message'},inplace=True)


# In[12]:


#df.head()


# In[13]:


#For duplicate values
#df.shape


# In[14]:


df.drop_duplicates(inplace=True)


# In[15]:


#df.shape


# In[16]:


df['labels']=df['labels'].map({'ham':0,'spam':1})


# In[17]:


#df.head()


# In[18]:


#remove stop words and punctuation marks
def clean_data(message):
    #removing punctuation and returning a list of remaining characters of message
    message_without_punc=[character for character in message if character not in string.punctuation]
    
    message_without_punc=''.join(message_without_punc)
    separator=' '
    return separator.join(word for word in message_without_punc.split() if word.lower() not in stopwords.words('english'))


# In[19]:


#applying the function to the dataset
df['message']=df['message'].apply(clean_data)


# In[20]:


#defining x and y variables
x=df['message']
y=df['labels']
#converting all words to vectors(basic step in any nlp model)
#cv=variable storing oblject of class CountVectorizer
cv = CountVectorizer()
x=cv.fit_transform(x)
#print(x)


# In[21]:


#split into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[22]:


#fitting data on model
model=MultinomialNB().fit(x_train,y_train)
predictions=model.predict(x_test)


# In[23]:


#print(accuracy_score(y_test,predictions))
#print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))


# In[24]:


#web application:
#function:
def predict(text):
    labels=['Not spam','Spam']
    x=cv.transform(text).toarray()
    p=model.predict(x)
    s=[str(i) for i in p]
    v=int(''.join(s))
    return str('This message is probably: '+labels[v])



