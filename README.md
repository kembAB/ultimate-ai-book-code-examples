# Ultimate AI Book — One-Question Code Examples

Two example programs from the Ultimate AI Book by Abel Haile Magicho.
Each part of the book ends with one question: write ONE program that uses
everything the part taught.

## Part 1 — Python for AI — part-1-python-for-ai.py

Questions 1 to 300 of the book. One program using all five pillars of
Part 1: variables, lists, functions, classes, and files.

It sets a passing mark of 60, defines an average function, builds a
Student class with a pass or fail result, averages the scores
72, 45, 91, 60 to get 67.0, creates two students — Abel with 72 and
Alicia with 91 — writes their results to results.txt, and reads the
file back.

Output:

    Average: 67.0
    Abel: Pass
    Alicia: Pass

Run it with: python3 part-1-python-for-ai.py

## Part 2 — Python AI Libraries — part-2-python-ai-libraries.py

Questions 301 to 600 of the book. One program using all four libraries
of Part 2: NumPy, pandas, Matplotlib, and scikit-learn.

It holds the scores 40, 55, 70, 85 in a NumPy array, prints the mean
62.5 and the top score 85, builds a pandas table of four students and
filters it to the scores of 60 or more — Alicia with 70 and Abel
with 85 — saves a bar chart as scores_chart.png, then trains a
LinearRegression model on study hours 1 to 4 and predicts the score
for 5 study hours: 100.0.

Output:

    Mean score: 62.5
    Top score: 85
    Passing rows: Alicia 70, Abel 85
    Predicted score for 5 study hours: [100.]

Needs numpy, pandas, matplotlib and scikit-learn installed.
Run it with: python3 part-2-python-ai-libraries.py
