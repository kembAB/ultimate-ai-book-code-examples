import random   # bring in the random tool: the toy model uses it to pick words when it writes new text
from collections import Counter   # bring in Counter: a box that counts how many times each thing appears
from sklearn.feature_extraction.text import TfidfVectorizer   # bring in the tool that turns text into lists of numbers
from sklearn.metrics.pairwise import cosine_similarity   # bring in the tool that measures how close two lists of numbers point

# THE TRAINING TEXT. A real LLM trains on trillions of words; this toy model trains on six sentences, to show the same ideas small.
sentences = ["the student studies hard and passes the exam",   # training sentence one
             "the student studies every day and improves",   # training sentence two
             "the student reads books and learns new words",   # training sentence three
             "the teacher asks questions and the student answers",   # training sentence four
             "the model reads text and predicts the next word",   # training sentence five
             "the model learns patterns and answers questions"]   # training sentence six

# STEP 1 — TOKENS: a model never reads whole sentences; it reads tokens, small pieces of text.
text = " ".join(sentences)   # join the six sentences into one long text
tokens = text.split()   # chop the text into tokens: in this toy, one token is one word
print("Tokens in the training text:", len(tokens))   # count every token the toy model can learn from
print("First tokens:", tokens[0:8])   # show the first eight tokens, so the chopping is visible
print("Different words it knows:", len(set(tokens)))   # the vocabulary: how many different words appear

# STEP 2 — TRAINING: count which word follows which. Those counts are everything the toy model learns.
followers = {}   # an empty notebook: for each word, the words that came right after it
for sentence in sentences:   # visit the training sentences one by one
    words = sentence.split()   # chop this sentence into its words
    for i in range(len(words) - 1):   # walk along the words, stopping one before the last
        word = words[i]   # the word we are standing on
        nxt = words[i + 1]   # the word that comes right after it
        if word not in followers:   # the first time this word is seen as a starting word...
            followers[word] = Counter()   # ...give it an empty counting box
        followers[word][nxt] += 1   # write down one more time that nxt followed word
print("After 'student' the counts are:", dict(followers["student"]))   # show what the toy model learned can follow 'student'

# STEP 3 — PREDICTION: the prompt is the text we hand the model; it predicts the next word from the counts.
prompt = "the student"   # the prompt: the text we give the model to continue
last_word = prompt.split()[-1]   # the model looks at the prompt's last word: its whole context here
counts = followers[last_word]   # look up what followed that word during training
total = sum(counts.values())   # add the counts up: how many times anything followed it
for nxt in counts:   # go through each word that ever followed it
    print("Chance the next word is", nxt + ":", round(counts[nxt] / total, 2))   # a count divided by the total is its chance
best = max(counts, key=counts.get)   # find the follower with the highest count
print("Most likely next word:", best)   # the model's single best guess for the next word

# STEP 4 — EMBEDDINGS: turn each sentence into a list of numbers, so that closeness can be measured.
vectorizer = TfidfVectorizer()   # create the tool that turns text into number lists
vectors = vectorizer.fit_transform(sentences)   # turn all six sentences into number lists, learning word weights on the way
print("Each sentence became a list of", vectors.shape[1], "numbers")   # one number for each different word in the text

# STEP 5 — RETRIEVAL: to answer a question, first find the training sentence closest to it.
question = "what does the student study every day"   # the question we ask; words it never saw are simply skipped
q_vector = vectorizer.transform([question])   # turn the question into a number list, the same way as the sentences
scores = cosine_similarity(q_vector, vectors)[0]   # measure the closeness of the question to every sentence
for i in range(len(sentences)):   # visit the sentences one by one
    print("Closeness to sentence", str(i + 1) + ":", round(float(scores[i]), 2))   # 1.0 means identical; 0.0 means nothing shared
best_i = int(scores.argmax())   # the position of the closest sentence
print("Closest sentence:", sentences[best_i])   # the sentence an answer would be built from

# STEP 6 — GENERATION: write new text one word at a time, picking each next word by chance.
def generate(start, temperature, seed, max_words=9):   # a small machine: a start word, a daring level, a fixed random start, a word budget
    random.seed(seed)   # fix the random picker's starting point, so this run always writes the same text
    words = [start]   # the text so far: just the starting word
    for _ in range(max_words - 1):   # keep adding words until the word budget is spent
        counts = followers.get(words[-1])   # look up what followed the current last word
        if not counts:   # if nothing ever followed it...
            break   # ...the text simply ends here
        options = list(counts.keys())   # the words that could come next
        weights = [counts[w] ** (1 / temperature) for w in options]   # temperature reshapes the chances: low sharpens, high flattens
        words.append(random.choices(options, weights=weights)[0])   # pick the next word, guided by the reshaped chances
    return " ".join(words)   # join the picked words back into one text

print("Low temperature (plays safe):", generate("the", 0.3, 13))   # temperature 0.3: the most learned words almost always win
print("High temperature (takes chances):", generate("the", 1.5, 13))   # temperature 1.5: weaker words get a real chance to appear
