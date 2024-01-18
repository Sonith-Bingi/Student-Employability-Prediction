from keras.layers import Dense, Conv1D, Flatten
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
import matplotlib.pyplot as plt

classfiers_sorted = [x for _, x in sorted(zip(acc, classifiers))]
acc = sorted(acc)

plt.figure().set_figwidth(15)
plt.plot(classfiers_sorted, acc)

# naming the x axis
plt.xlabel('CLASSFIERS')
# naming the y axis
plt.ylabel('ACCURACY')

# DEEPLEARNING
# READING THE DATASET

X = df[["AddnlCourses?", "StudyBlogs?", "WriteRsrchPapers?", "ListenTechVideos?", "CourseCertifications?",
        "IntnlConfattended?", "PublishedRsrchPapers?", "PresentedInConferences?", "WroteTechBlogs?", "DevelopedTechVideos?"]]

y = df["Employability?"].values.reshape(-1, 1)
# END

X = X.values.reshape(X.shape[0], X.shape[1], 1)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


model = Sequential()
model.add(Conv1D(128, 2, activation='relu', input_shape=(10, 1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

print(model.summary())

history = model.fit(X_train, y_train, batch_size=100, epochs=500, verbose=0)
y_pred = model.predict(X_test)

print(model.evaluate(X_train, y_train))
print('MSE: %.4f' % mean_squared_error(y_test, y_pred))

fig = plt.figure(figsize=(15, 5))
x_ax = range(len(y_pred))
plt.scatter(x_ax, y_test, s=5, color='blue', label='original')
plt.plot(x_ax, y_pred, lw=0.8, color='red', label='predicted')
plt.legend()
plt.show()

y_pred1 = y_pred.flatten()
y_predlist = y_pred1.tolist()
print(y_predlist)

for i in range(len(y_predlist)):
    if y_predlist[i] > 1.5:
        y_predlist[i] = 2
    elif y_predlist[i] <= 0.5:
        y_predlist[i] = 0
    else:
        y_predlist[i] = 1

print(y_predlist)

y_test1 = y_test.flatten()
y_testlist = y_test1.tolist()
print(y_testlist)

truepred = 0
falsepred = 0

for i in range(len(y_predlist)):
    if y_predlist[i] == y_testlist[i]:
        truepred = truepred+1
    else:
        falsepred = falsepred+1

accuracydl = truepred/(truepred+falsepred)

print(accuracydl)
