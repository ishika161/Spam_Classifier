#IMPORT LIBRARIES
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


#GET THE DATASET
df=pd.read_csv('emails.csv')
df.head()

df.info()


ham=df[df['spam']==0]
spam=df[df['spam']==1]

#GET DISTRIBUTION PERCENTAGE
print(f"Spam % ={(len(spam)/len(df))*100}")
print(f"Ham % ={(len(ham)/len(df))*100}")

#VISUALISE DATA
sns.countplot(df['spam'],label="count spam vs ham")

#APPLY VECTORIZER 
from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()
count=v.fit_transform(df['text'])
print(v.get_feature_names())
print(count.toarray())
count.shape

label=df['spam'].values


#SPLIT DATA
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(count,label,test_size=0.2)

#TRAIN THE MODEL
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(x_train,y_train)

#TEST 1
test=['money matters']
tcount=v.transform(test)
tcount.shape
pred=nb.predict(tcount)

 
#TEST 
testing_sample = ["Hi Joe,you are our lucky winner"]
testing_sample_countvectorizer = v.transform(testing_sample)
test_predict = nb.predict(testing_sample_countvectorizer)
test_predict

if test_predict[0]==1:
    print("It is a Spam Message.")
else:
    print("It is not a Spam Message.")


#evaluate
from sklearn.metrics import classification_report,confusion_matrix
y_pred=nb.predict(x_train)
cm=confusion_matrix(y_train,y_pred)
sns.heatmap(cm,annot=True)


y_pred2=nb.predict(x_test)
cm2=confusion_matrix(y_test,y_pred2)
sns.heatmap(cm2,annot=True)

print(classification_report(y_test,y_pred2))