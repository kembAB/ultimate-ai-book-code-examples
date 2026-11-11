import numpy as np   # bring in NumPy and give it the short name np
from sklearn.model_selection import train_test_split   # bring in the tool that splits data into a training set and a test set
from sklearn.linear_model import LogisticRegression   # bring in one ready-made classification model

# The data: 16 students. Each row holds two clues: hours studied and practice tests taken.
students = np.array([[1, 0], [2, 0], [2, 1], [3, 1], [3, 2], [4, 1], [4, 2], [5, 2],
                     [5, 3], [6, 2], [6, 3], [7, 3], [7, 4], [8, 3], [8, 4], [9, 4]])   # the clues the model learns from
results = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])   # the answers: 0 means failed, 1 means passed

# The split: hold back some students as a test the model has never seen.
X_train, X_test, y_train, y_test = train_test_split(students, results, test_size=0.25, random_state=1)   # a fixed shuffle, so the split lands the same way every run: 12 students for training, 4 kept hidden for the test

# Training: the model studies the training students and learns the pattern by itself.
model = LogisticRegression(max_iter=1000)   # create the model, still empty and untrained
model.fit(X_train, y_train)   # train it: show it the clues together with the real answers until it finds the pattern

# Prediction: ask the trained model about a brand-new student.
print("Prediction for a new student with 6 hours and 3 practice tests:", model.predict([[6, 3]]))   # the model answers 1, which means passed

# Evaluation: check the model on the hidden test students.
print("Accuracy on the test students:", model.score(X_test, y_test))   # the share of test students the model judged correctly

# Why models fail: train the same model on wrong answers, and it learns them faithfully.
broken_model = LogisticRegression(max_iter=1000)   # create a second model, exactly the same kind
broken_model.fit(X_train, 1 - y_train)   # train it on flipped answers: every pass marked fail, every fail marked pass
print("Accuracy of the model trained on wrong answers:", broken_model.score(X_test, y_test))   # tested on the real answers, it fails
