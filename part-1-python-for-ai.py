passing_mark = 60                 # remember the score needed to pass, and call it passing_mark

def average(scores):         # build a reusable machine named average; it receives a list of scores
    return sum(scores) / len(scores)   # add all scores, divide by how many there are, hand back the result

class Student:               # a blueprint for making student objects, like a mold
    def __init__(self, name, score):   # runs when a new student is created; stores their name and score
        self.name = name     # keep this student's name on the student itself
        self.score = score   # keep this student's score on the student itself
    def result(self):        # a built-in action every student can do: report pass or fail
        if self.score >= passing_mark:   # if the score reaches the passing mark...
            return "Pass"    # ...the answer is Pass
        return "Fail"        # otherwise the answer is Fail

scores = [72, 45, 91, 60]    # one list holding four scores in order
print("Average:", average(scores))   # run the average machine on the list and show the result: 67.0

students = [Student("Abel", 72), Student("Alicia", 91)]            # use the blueprint twice to make two real students

with open("results.txt", "w") as f:   # open a file for writing; "with" closes it safely when done
    for s in students:       # go through the students one by one
        f.write(s.name + ": " + s.result() + "\n")   # write name, then the pass/fail answer, then a new line

with open("results.txt", "r") as f:   # open the same file again, this time for reading
    print(f.read())          # read everything in the file and show it on screen
