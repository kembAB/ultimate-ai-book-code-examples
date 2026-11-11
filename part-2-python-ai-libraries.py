import numpy as np           # bring in NumPy and give it the short name np
import pandas as pd          # bring in pandas and give it the short name pd
import matplotlib.pyplot as plt   # bring in the chart maker and call it plt
from sklearn.linear_model import LinearRegression   # bring in one ready-made learning model

scores = np.array([40, 55, 70, 85])        # a NumPy array: one fast table of numbers, all the same type
print("Mean score:", scores.mean())        # NumPy does the math on the whole table at once: the mean is 62.5
print("Top score:", scores.max())          # the biggest number in the table: 85

students = pd.DataFrame({"Name": ["Sara", "John", "Alicia", "Abel"], "Score": scores})   # a pandas table: names in one column, scores in the other
print(students[students["Score"] >= 60])   # show only the rows where the score reaches 60: Alicia and Abel

plt.bar(students["Name"], students["Score"])   # draw one bar per student, as tall as the score
plt.savefig("scores_chart.png")   # save the chart as a picture file instead of only showing it

hours = np.array([[1], [2], [3], [4]])   # study hours: the clues the model learns from, one row per student
model = LinearRegression()   # create the model, still empty and untrained
model.fit(hours, scores)     # train it: show it the hours together with the real scores until it finds the pattern
print("Predicted score for 5 study hours:", model.predict([[5]]))   # ask the trained model to guess the score for 5 hours: 100.0
