import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re

df = pd.read_csv("./datasets/train_tweets.csv")
df.head()

hate_tweet = df[df.label == 1]
hate_tweet.head()

normal_tweet = df[df.label == 0]
normal_tweet.head()

# Hate Word clouds
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text = " ".join(review for review in hate_tweet.tweet)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
fig = plt.figure(figsize = (20, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#distributions
df_Stat=df[['label','tweet']].groupby('label').count().reset_index()
df_Stat.columns=['label','count']
df_Stat['percentage']=(df_Stat['count']/df_Stat['count'].sum())*100
df_Stat

def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",tweet.lower()).split())

df['processed_tweets'] = df['tweet'].apply(process_tweet)
df.head()

#As this dataset is highly imbalance we have to balance this by over sampling
cnt_non_fraud = df[df['label'] == 0]['processed_tweets'].count()
df_class_fraud = df[df['label'] == 1]
df_class_nonfraud = df[df['label'] == 0]
df_class_fraud_oversample = df_class_fraud.sample(cnt_non_fraud, replace=True)
df_oversampled = pd.concat([df_class_nonfraud, df_class_fraud_oversample], axis=0)

print('Random over-sampling:')
print(df_oversampled['label'].value_counts())

# Split data into training and test sets
from sklearn.model_selection import train_test_split
X = df_oversampled['processed_tweets']
y = df_oversampled['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = None)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)
#%%
x_train_counts = count_vect.fit_transform(X_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)
#%%
print(x_train_counts.shape)
print(x_train_tfidf.shape)

x_test_counts = count_vect.transform(X_test)
x_test_tfidf = transformer.transform(x_test_counts)
#%%
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=500)
model.fit(x_train_tfidf,y_train)

predictions = model.predict(x_test_tfidf)
#%%

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))

#Building XGBoost Model
import xgboost as xgb
model_bow = xgb.XGBClassifier(random_state=22,learning_rate=0.9)
model_bow.fit(x_train_tfidf,y_train)

predict_xgb = model_bow.predict(x_test_tfidf)
#%%
print(confusion_matrix(y_test,predict_xgb))
print(classification_report(y_test,predict_xgb))

#SVM Model
from sklearn import svm
lin_clf = svm.LinearSVC()
lin_clf.fit(x_train_tfidf,y_train)

predict_svm = lin_clf.predict(x_test_tfidf)
#%%
from sklearn.metrics import confusion_matrix,f1_score
print(confusion_matrix(y_test,predict_svm))
print(classification_report(y_test, predict_svm))

#Logistic Regression Model
from sklearn.linear_model import LogisticRegression
#%%
logreg = LogisticRegression(random_state=42)
#%%
#Building Logistic Regression  Model
logreg.fit(x_train_tfidf,y_train)

predict_log = logreg.predict(x_test_tfidf)
#%%
print(confusion_matrix(y_test,predict_log))
print(classification_report(y_test, predict_log))

from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(x_train_tfidf, y_train)

predict_nb = NB.predict(x_test_tfidf)
#%%
print(confusion_matrix(y_test,predict_nb))
print(classification_report(y_test, predict_nb))

#Test Data Set
df_test = pd.read_csv("/datasets/test_tweets.csv")
df_test.head()

df_test['processed_tweets'] = df_test['tweet'].apply(process_tweet)
df_test.head()

X = df_test['processed_tweets']
x_test_counts = count_vect.transform(X)
x_test_tfidf = transformer.transform(x_test_counts)
#%%
df_test['predict_nb'] = NB.predict(x_test_tfidf)
df_test[df_test['predict_nb']==1]

df_test['predict_svm'] = NB.predict(x_test_tfidf)
df_test['predict_rf'] = model.predict(x_test_tfidf)
df_test.head()

file_name = 'test_predictions_Twitter Hate Analysis.csv'
df_test.to_csv(file_name,index=False)