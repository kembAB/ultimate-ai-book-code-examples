import numpy as np   # bring in NumPy and give it the short name np
from sklearn.feature_extraction.text import TfidfVectorizer   # bring in TF-IDF: the tool that turns text into numbers a model can read
from sklearn.model_selection import train_test_split   # bring in the tool that splits data into a training set and a test set
from sklearn.linear_model import LogisticRegression   # bring in a classification model: it predicts a category
from sklearn.metrics.pairwise import cosine_similarity   # bring in the tool that measures how alike two rows of numbers are

# THE RAW TEXT — 16 short student feedback sentences about a class. A computer cannot read words, only numbers, so the words must be washed and counted first.
feedback = ["The teacher makes the lessons clear and helps every student",   # positive: clear, helps
            "The lessons are easy and fun and the teacher is kind",   # positive: easy, fun, kind
            "The homework is helpful and clear and the teacher is patient and kind",   # positive: helpful, clear, kind
            "Great lessons, the class is fun and the teacher is excellent",   # positive: great, fun, excellent
            "I love the lessons because the teacher explains clearly and helps us",   # positive: love, helps
            "The course is excellent, the lessons are easy and the homework is fun",   # positive: excellent, easy, fun
            "Happy students learn fast because the teacher is helpful and clear",   # positive: happy, helpful, clear
            "The teacher is great and kind and the lessons are fun and easy",   # positive: great, kind, fun, easy
            "The lessons are boring and hard and the teacher never explains",   # negative: boring, hard
            "The homework is confusing and useless and the teacher is rude",   # negative: confusing, useless, rude
            "Terrible lessons, the class is boring and the tests are unfair",   # negative: terrible, boring, unfair
            "I hate the lessons because the teacher is slow and confusing",   # negative: hate, slow, confusing
            "The course is awful, the lessons are hard and the homework is useless",   # negative: awful, hard, useless
            "Bad teacher, the lessons are confusing and the class is boring",   # negative: bad, confusing, boring
            "The teacher is rude and slow and the lessons are terrible",   # negative: rude, slow, terrible
            "The homework is hard and unfair and the teacher is unkind"]   # negative: hard, unfair, unkind
labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])   # the sentiment of each sentence: 1 means positive, 0 means negative

# THE WASHING LINE — stop words are little glue words that carry almost no feeling, so they are dropped before counting.
stopwords = {"the", "is", "are", "and", "i", "this", "in", "to", "a", "of", "because", "every", "us"}   # the filler words to throw away: the, is, are, and...

def stem(word):   # a tiny stemmer: it cuts a word down to its root form, so "lessons" and "lesson" count as the same word
    for ending in ("ing", "ed"):   # first try the longer endings: -ing and -ed
        if word.endswith(ending) and len(word) > len(ending) + 2:   # cut only when a real root stays behind
            return word[:-len(ending)]   # hand back the word with the ending cut off
    if word.endswith("s") and len(word) > 3:   # then try a trailing -s, the usual mark of a plural
        return word[:-1]   # hand back the word without the final s
    return word   # no ending matched, so the word is already its own root

def clean(text):   # a small machine that washes one sentence and hands back its useful words
    words = text.lower().split()   # lowercase every letter, then tokenize: split the sentence into single words
    useful = []   # an empty basket where the cleaned words will collect
    for word in words:   # look at the words one at a time
        word = word.strip(",.!?")   # cut any punctuation off the edges of the word
        if word not in stopwords and word != "":   # keep the word only if it is not a filler word and not empty
            useful.append(stem(word))   # cut the kept word down to its root and drop it in the basket
    return useful   # hand back the basket of cleaned root words

print("First sentence before cleaning:", feedback[0])   # the raw sentence, exactly as the student wrote it
print("First sentence after cleaning:", clean(feedback[0]))   # the same sentence washed: lowercase tokens, no stop words, roots only
cleaned = [" ".join(clean(sentence)) for sentence in feedback]   # wash all 16 sentences and join each one's roots back into a string

# TF-IDF — turn each washed sentence into a row of numbers: words in every sentence weigh little, rare telling words weigh a lot.
vectorizer = TfidfVectorizer()   # create the TF-IDF tool, still empty and untrained
vectors = vectorizer.fit_transform(cleaned)   # learn the vocabulary from the 16 sentences and turn each sentence into numbers
print("Words in the vocabulary:", len(vectorizer.get_feature_names_out()))   # how many different root words the tool found
print("Shape of the number table:", vectors.shape)   # 16 rows, one per sentence, and one column per vocabulary word

# COSINE SIMILARITY — sentences with the same feeling point the same way, so the angle between their number rows is small.
print("Similarity, sentence 1 and sentence 7 (both positive):", round(cosine_similarity(vectors[0], vectors[6])[0][0], 2))   # both praise the teacher, so their rows point together
print("Similarity, sentence 1 and sentence 12 (positive vs negative):", round(cosine_similarity(vectors[0], vectors[11])[0][0], 2))   # opposite feelings, so their rows point apart

# TRAIN THE SENTIMENT CLASSIFIER — split first, then let the model learn which number patterns mean praise and which mean complaint.
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.25, random_state=7)   # a fixed shuffle: 12 sentences train, 4 stay hidden
model = LogisticRegression(max_iter=1000)   # create the classification model, still empty and untrained
model.fit(X_train, y_train)   # train it: it learns how much weight each word's number carries toward positive or negative
print("Accuracy on the test sentences:", model.score(X_test, y_test))   # the share of hidden sentences it sorted into the right feeling

# ONE BRAND-NEW SENTENCE — wash it the same way, turn it into numbers, and ask the model how it feels.
new_sentence = "The teacher is kind and helpful and the lessons are fun"   # a new sentence the model has never seen
print("New sentence after cleaning:", clean(new_sentence))   # the new sentence washed by the same machine
new_vector = vectorizer.transform([" ".join(clean(new_sentence))])   # turn it into a row of numbers with the same vocabulary
names = ["negative", "positive"]   # the two answer words, in label order: 0 first, then 1
print("Prediction for the new sentence:", names[model.predict(new_vector)[0]])   # the model's verdict, printed as a word
