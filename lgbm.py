# LGBMClassifier
import sklearn
from lightgbm import LGBMClassifier

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
