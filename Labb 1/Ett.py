import sklearn as sk
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')

from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


#Loading the data
trainData = pd.read_csv("lab_train.txt")
testData = pd.read_csv("lab_test.txt")


#Vectorizing and removing stopwords
stopset=set(nltk.stopwords.words('english'))
vectorizer = text.TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
vectorizedData = vectorizer.fit_transform(trainData.review)
vectorizedData.shape


#Labels
labels=[]
for d in trainData.score:
    if(d < 3):
        labels.append('negative')
    else:
        labels.append('positive')


#Train/test-split
reviewTrain, reviewTest, labelTrain, labelTest = sk.model_selection.train_test_split(vectorizedData,labels, random_state=42, test_size=0.33)


#Training the classifier and calculating its accuracy
Bayes = MultinomialNB().fit(reviewTrain, labelTrain)

truth = np.array(labelTest)
train_test=np.array(Bayes.predict(reviewTest))
result = accuracy_score(truth, train_test)


print(Bayes.predict(reviewTest))
print(result)
print(Bayes.predict(testData.review))





