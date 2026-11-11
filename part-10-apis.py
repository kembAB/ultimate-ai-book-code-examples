import numpy as np   # bring in NumPy and give it the short name np
from sklearn.linear_model import LogisticRegression   # bring in a classification model: it sorts students into pass or fail
from fastapi import FastAPI   # bring in FastAPI, the framework that turns Python functions into a web API
from fastapi.testclient import TestClient   # bring in a pretend visitor that can call the API from inside this same program
from pydantic import BaseModel   # bring in the form-maker: it describes exactly what data a request must carry
# STEP 1 — Train a tiny model first: the pass predictor from Parts 3 and 4, this time trained on all 16 students.
students = np.array([[1, 0], [2, 0], [2, 1], [3, 1], [3, 2], [4, 1], [4, 2], [5, 2],
                     [5, 3], [6, 2], [6, 3], [7, 3], [7, 4], [8, 3], [8, 4], [9, 4]])   # the clues: hours studied and practice tests taken
passed = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])   # the label for each student: 0 means failed, 1 means passed
model = LogisticRegression(max_iter=1000)   # create the classification model, still empty and untrained
model.fit(students, passed)   # train it on all sixteen students: more study and more practice tests mean more likely to pass

# STEP 2 — Build the API: one app object, with a name and a plain description the automatic docs will show.
app = FastAPI(title="Pass Predictor API", description="Predicts pass or fail from hours and practice tests")   # create the API: endpoints hang off it
# STEP 3 — Describe the request body: a form with two boxes that every prediction request must fill in.
class StudentIn(BaseModel):   # a form named StudentIn: any request to /predict must match this shape, or it is refused
    hours: float   # first box: hours studied, a number (a decimal like 6.5 is allowed)
    practice_tests: int   # second box: practice tests taken, a whole number

# STEP 4 — The /health endpoint: a GET request asks a question, and this function answers it.
@app.get("/health")   # whenever a GET request arrives at the address /health, run the function below
def health():   # the health check: the simplest possible job, just proving the API is alive
    return {"status": "ok"}   # the response body, sent back as JSON: the API is up and answering
# STEP 5 — The /predict endpoint: a POST request carries a student in, and the model answers.
@app.post("/predict")   # whenever a POST request arrives at the address /predict, run the function below
def predict(student: StudentIn):   # the request body is checked against the StudentIn form before this function runs
    result = model.predict([[student.hours, student.practice_tests]])[0]   # ask the trained model about this student: 0 means fail, 1 means pass
    word = "Pass" if result == 1 else "Fail"   # turn the model's number into a plain word a person can read
    return {"prediction": int(result), "result": word}   # the response body, sent back as JSON: the number and the word together

# STEP 6 — Call the API from inside this program: the TestClient plays the visitor, no real server needed.
client = TestClient(app)   # create the pretend visitor, wired straight to this app
health_reply = client.get("/health")   # send a GET request to /health, asking: are you alive?
print("Health check status code:", health_reply.status_code)   # the status code that came back: 200 means the request worked
print("Health check response:", health_reply.json())   # the JSON body the API sent back
predict_reply = client.post("/predict", json={"hours": 6, "practice_tests": 3})   # send a POST request to /predict, carrying one student's data as JSON
print("Predict status code:", predict_reply.status_code)   # the status code that came back: 200 means the prediction worked
print("Predict response:", predict_reply.json())   # the model's answer, as JSON: 1 and Pass for this student
bad_reply = client.post("/predict", json={"hours": 6})   # send a broken POST request: the practice_tests box is missing
print("Broken request status code:", bad_reply.status_code)   # 422 means the API refused it: the request body failed the form check
docs_reply = client.get("/docs")   # ask for the automatic documentation page that FastAPI builds for this API
print("Docs page status code:", docs_reply.status_code)   # 200 means the free docs page exists and answers
