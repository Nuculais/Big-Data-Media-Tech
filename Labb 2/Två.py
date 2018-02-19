import sklearn as sk
from sklearn.feature_extraction import text
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


#Loading the data
trainData = pd.read_csv("lab_train.txt")
testData = pd.read_csv("lab_test.txt")


#Vectorizing and removing stopwords
vectorizer = text.TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words='english')
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
reviewTrain, reviewTest, labelTrain, labelTest = sk.model_selection.train_test_split(vectorizedData, labels, random_state=42, test_size=0.3)

#reviewTrain.shape
#labelTrain.shape
#reviewTest.shape
#labelTest.shape
#'list' object has no attribute 'shape'

SupportVectorMachine = sk.svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

SupportVectorMachine.fit(reviewTrain, labelTrain)


result = SupportVectorMachine.predict(reviewTest)
trueLabels = np.array(labelTest)
theAccuracy = accuracy_score(trueLabels, result)

print(result)
print(theAccuracy)