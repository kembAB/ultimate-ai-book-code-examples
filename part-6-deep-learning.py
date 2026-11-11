import numpy as np   # bring in NumPy and give it the short name np
from sklearn.model_selection import train_test_split   # bring in the tool that splits data into a training set and a test set
from sklearn.preprocessing import StandardScaler   # bring in the tool that rescales numbers so big and small clues count fairly
from sklearn.neural_network import MLPClassifier   # bring in a real neural network: layers of neurons that learn from examples

# THE DATA — the same 16 students as Part 4: hours studied and practice tests taken, and whether each passed.
students = np.array([[1, 0], [2, 1], [3, 1], [2, 2], [4, 1], [3, 3], [5, 1], [4, 3],
                     [5, 2], [6, 2], [5, 3], [7, 2], [6, 4], [8, 3], [7, 4], [9, 3]])   # the clues: hours studied and practice tests taken
passed = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])   # the label: 0 means failed, 1 means passed

# Split the data: most students for training, a few hidden for the test.
X_train, X_test, y_train, y_test = train_test_split(students, passed, test_size=0.25, random_state=42)   # a fixed shuffle: 12 students train, 4 stay hidden

# Scale the clues: hours run 1 to 9, practice tests 0 to 4 — rescaling puts both clues on the same footing, and networks learn far better that way.
scaler = StandardScaler()   # create the scaler, still unused
X_train = scaler.fit_transform(X_train)   # learn the scale from the training students, then rescale them
X_test = scaler.transform(X_test)   # rescale the hidden students with the same scale

# BUILD THE NETWORK — neurons in layers. 2 input neurons (one per clue), two hidden layers of 8 and 4 neurons, then 1 output neuron that answers pass or fail. More than one hidden layer is what makes it deep.
network = MLPClassifier(hidden_layer_sizes=(8, 4), activation="relu", solver="adam", max_iter=1000, random_state=1)   # relu is the activation: each neuron passes on positive signals and blocks negative ones; adam is the optimiser that adjusts the weights

# TRAIN IT — each epoch is one full pass through the training students: a forward pass makes guesses, the loss measures how wrong they are, and backpropagation sends the blame backwards so the optimiser can nudge every weight.
network.fit(X_train, y_train)   # train the network on the 12 training students

# Watch the loss fall: the loss curve holds one loss value for every epoch of training.
print("Epochs trained:", len(network.loss_curve_))   # how many full passes through the training data the network needed
print("Loss at the first epoch:", round(network.loss_curve_[0], 4))   # how wrong the network was at the very start
print("Loss at the final epoch:", round(network.loss_curve_[-1], 4))   # how wrong it was by the end — much lower means it learned

# CHECK FOR OVERFITTING — compare the students it trained on with the students it never saw.
print("Accuracy on the training students:", network.score(X_train, y_train))   # a perfect score here alone proves nothing: it may have memorised them
print("Accuracy on the test students:", network.score(X_test, y_test))   # the honest score: students the network never saw while learning

# ONE NEW STUDENT.
print("Prediction for a student with 7 hours and 3 practice tests:", network.predict(scaler.transform([[7, 3]])))   # rescale the new student the same way, then ask the network: 1 means passed
