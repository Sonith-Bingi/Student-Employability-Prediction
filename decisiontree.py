# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# LOADING INTO CLASSIFIER
classifier = DecisionTreeClassifier()
classifier.fit(training_inputs, training_outputs)
predictions = classifier.predict(testing_inputs)
accuracy = 100.0 * accuracy_score(testing_outputs, predictions)
precision = 100.0 * \
    precision_score(testing_outputs, predictions, average='weighted')
recall = 100.0 * recall_score(testing_outputs, predictions, average='weighted')
f1score = 100.0 * f1_score(testing_outputs, predictions, average='weighted')

# RESULTS OF THE CLASSIFIER
print("Accuracy of Decision Tree: " + str(accuracy))
print("Precision of Decision Tree: " + str(precision))
print("Recall of Decision Tree: " + str(recall))
print("F1 Score of Decision Tree: " + str(f1score))

acc.append(accuracy)
prec.append(precision)
rec.append(recall)
f1.append(f1score)
