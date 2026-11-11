import numpy as np   # bring in NumPy and give it the short name np
from sklearn.model_selection import train_test_split   # bring in the tool that splits data into a training set and a test set
from sklearn.linear_model import LinearRegression   # bring in a regression model: it predicts a number
from sklearn.tree import DecisionTreeClassifier   # bring in a classification model: it predicts a category

# REGRESSION — predicting a NUMBER. The data: 12 students, hours studied and the exam score each one got.
hours = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]])   # the clue: hours studied
scores = np.array([46, 51, 55, 61, 64, 70, 76, 79, 85, 91, 94, 99])   # the label is a number: the exam score

# Split the regression data: most students for training, a few hidden for the test.
h_train, h_test, s_train, s_test = train_test_split(hours, scores, test_size=0.25, random_state=1)   # a fixed shuffle: 9 students train, 3 stay hidden

# Train the regression model on the training students.
score_model = LinearRegression()   # create the regression model, still empty and untrained
score_model.fit(h_train, s_train)   # train it: it finds the line that best matches hours to scores

# Check the regression model on the hidden students, and measure its average error.
guesses = score_model.predict(h_test)   # the model's predicted scores for the hidden students
print("Predicted scores:", np.round(guesses, 1))   # what the model guessed, rounded to one decimal
print("Real scores:", s_test)   # what those students really scored
print("Average error:", round(np.mean(np.abs(guesses - s_test)), 2))   # on average, how many points each guess missed by

# CLASSIFICATION — predicting a CATEGORY. The data: 16 students, hours and practice tests, and whether each passed.
students = np.array([[1, 0], [2, 1], [3, 1], [2, 2], [4, 1], [3, 3], [5, 1], [4, 3],
                     [5, 2], [6, 2], [5, 3], [7, 2], [6, 4], [8, 3], [7, 4], [9, 3]])   # the clues: hours studied and practice tests taken
passed = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])   # the label is a category: 0 means failed, 1 means passed

# Split the classification data the same way.
X_train, X_test, y_train, y_test = train_test_split(students, passed, test_size=0.25, random_state=42)   # 12 students train, 4 stay hidden

# Train the classification model: a decision tree, which learns a chain of yes-or-no questions.
pass_model = DecisionTreeClassifier()   # create the classification model, still empty and untrained
pass_model.fit(X_train, y_train)   # train it: it learns which questions about hours and practice separate pass from fail

# Check the classification model on its hidden students.
print("Accuracy on the test students:", pass_model.score(X_test, y_test))   # the share of hidden students it put in the right category

# ONE NEW STUDENT, BOTH KINDS OF ANSWER.
print("Predicted score for 11 hours of study:", round(score_model.predict([[11]])[0], 1))   # regression answers with a number
print("Prediction for a student with 7 hours and 3 practice tests:", pass_model.predict([[7, 3]]))   # classification answers with a category: 1 means passed
