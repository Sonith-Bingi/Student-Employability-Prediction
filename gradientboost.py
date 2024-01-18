# IMPORTING NUMPY AND PANDAS
from lightgbm import LGBMClassifier
import sklearn
import numpy as np
import pandas as pd

# IMPORTING SKLEARN
from sklearn import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# READING THE DATASET
df = pd.read_csv('eduset.csv')

# df=df.loc[0:1000]
# MAPPING THE DATASET INTO NUMERIC VALUES
df["AddnlCourses?"] = df["AddnlCourses?"].map(
    {'NoCourses': -1, 'LessThan3': 0, 'ThreeOrmore': 1})
df["StudyBlogs?"] = df["StudyBlogs?"].map({'StudyBlogs': 1, 'DoNotStudy': 0})
df["WriteRsrchPapers?"] = df["WriteRsrchPapers?"].map(
    {'WriteRsrchPapers': 1, 'NoRsrchPapers': 0})
df["ListenTechVideos?"] = df["ListenTechVideos?"].map(
    {'ListenTechVideos': 1, 'DoNotListen': 0})
df["CourseCertifications?"] = df["CourseCertifications?"].map(
    {'CourseCertifications': 1, 'NoCourseCertifications': 0})
df["IntnlConfattended?"] = df["IntnlConfattended?"].map(
    {'AttendedIntnlConf': 1, 'NoIntnlConfs': 0})
df["PublishedRsrchPapers?"] = df["PublishedRsrchPapers?"].map(
    {'NoRsrchPapers': -1, 'National': 0, 'International': 1})
df["PresentedInConferences?"] = df["PresentedInConferences?"].map(
    {'PresentedInConferences': 1, 'NotPresentedInConferences': 0})
df["WroteTechBlogs?"] = df["WroteTechBlogs?"].map(
    {'WroteTechBlogs': 1, 'NoTechBlogs': 0})
df["DevelopedTechVideos?"] = df["DevelopedTechVideos?"].map(
    {'DevelopedTechVideos': 1, 'NoTechVideos': 0})
df["Employability?"] = df["Employability?"].map(
    {'Good': 2, 'Reasonable': 1, 'Low': 0})
data = df[["AddnlCourses?", "StudyBlogs?", "WriteRsrchPapers?", "ListenTechVideos?", "CourseCertifications?", "IntnlConfattended?",
           "PublishedRsrchPapers?", "PresentedInConferences?", "WroteTechBlogs?", "DevelopedTechVideos?", "Employability?"]].to_numpy()


inputs = data[:, :-1]
outputs = data[:, -1]
training_inputs, testing_inputs, training_outputs, testing_outputs = train_test_split(
    inputs, outputs, test_size=0.2, random_state=None, shuffle=True)
# inputs = data[:,:-1]
# outputs = data[:, -1]
# training_inputs = inputs[:100]
# training_outputs = outputs[:100]
# testing_inputs = inputs[100:]
# testing_outputs = outputs[100:]

classifiers = ['LGBM', 'AdaBoost', 'Gradient Boosting', 'CATBoost', 'KNN',
               'Support Vector', 'Decision Tree', 'Logistic Regression', 'Naive Bayes']
acc = []
prec = []
rec = []
f1 = []


# LGBMClassifier

# LOADING INTO CLASSIFIER
classifier = LGBMClassifier()
classifier.fit(training_inputs, training_outputs)
predictions = classifier.predict(testing_inputs)
accuracy = 100.0 * accuracy_score(testing_outputs, predictions)
precision = 100.0 * \
    precision_score(testing_outputs, predictions, average='weighted')
recall = 100.0 * recall_score(testing_outputs, predictions, average='weighted')
f1score = 100.0 * f1_score(testing_outputs, predictions, average='weighted')

# RESULTS OF THE CLASSIFIER
print("Accuracy of LGBMClassifier: " + str(accuracy))
print("Precision of LGBMClassifier: " + str(precision))
print("Recall of LGBMClassifier: " + str(recall))
print("F1 Score of LGBMClassifier: " + str(f1score))

acc.append(accuracy)
prec.append(precision)
rec.append(recall)
f1.append(f1score)

# testSet = [[0,0,1,0,0,0,0,0,0,0]]
# test = pd.DataFrame(testSet)
# predictions = classifier.predict(test)
# print('1: ',predictions)

# testSet = [[0,0,1,0,1,1,1,0,0,1]]
# test = pd.DataFrame(testSet)
# predictions = classifier.predict(test)
# print('2: ',predictions)

# testSet = [[1,1,1,0,1,1,0,1,1,1]]
# test = pd.DataFrame(testSet)
# predictions = classifier.predict(test)
# print('3: ',predictions)
