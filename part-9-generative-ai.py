import random   # bring in Python's random tools — the dice this generator rolls every time it picks a word
import numpy as np   # bring in NumPy and give it the short name np — fast number grids for the generated picture
import matplotlib.pyplot as plt   # bring in the drawing tool that saves the generated picture as a PNG file

# A SMALL GENERATIVE AI. Big generative models learn patterns from millions of examples, then create
# NEW content that follows those patterns. This toy does the same job in miniature: it learns from nine
# short sentences, writes brand-new sentences of its own, and finishes by generating a small picture.

# STEP 1 — THE TRAINING TEXT: the examples the generator learns from.
sentences = ["the robot paints a bright star in the night sky.",   # training sentence 1
             "the robot paints a bright moon in the night sky.",   # training sentence 2
             "the robot draws a small comet in the dark sky.",   # training sentence 3
             "the artist paints a bright star over the quiet sea.",   # training sentence 4
             "the artist draws a small moon over the quiet sea.",   # training sentence 5
             "a bright star shines in the night sky.",   # training sentence 6
             "a small moon shines over the quiet sea.",   # training sentence 7
             "the painter mixes bright drops of paint for the night sky.",   # training sentence 8
             "the painter paints a bright star in the night sky."]   # training sentence 9
training_text = " ".join(sentences)   # join the nine sentences into one training text

# STEP 2 — TOKENS AND THE CHAIN: cutting the text up and learning what follows what.
tokens = training_text.split()   # cut the text into tokens: one token per word, the full stop stays attached
print("Tokens in the training text:", len(tokens))   # how many tokens the model trained on
print("Different tokens the model knows:", len(set(tokens)))   # its vocabulary: the different words it can use

chain = {}   # the chain starts empty: it will link each pair of words to the words that came next
for i in range(len(tokens) - 2):   # walk through the text, looking at three tokens at a time
    pair = (tokens[i], tokens[i + 1])   # the current pair of words
    next_word = tokens[i + 2]   # the word that really followed that pair in the training text
    if pair not in chain:   # the first time this pair is seen...
        chain[pair] = []   # ...give it an empty list of follower words
    chain[pair].append(next_word)   # record that this word followed this pair once
print("Word pairs the model learned:", len(chain))   # how many pairs it can continue from

# STEP 3 — THE PROMPT: the starting words we hand the generator.
prompt = ("the", "robot")   # the prompt: the first two words of every sentence it writes
followers = chain[prompt]   # every word that followed this pair in the training text
print("After the prompt 'the robot', the model saw:", followers)   # the choices it holds for word three

# STEP 4 — SAMPLING WITH TEMPERATURE: how the next word gets chosen.
def pick_next(pair, temperature):   # define a tool: choose the next word after a pair of words
    options = chain[pair]   # the words that followed this pair, one entry per sighting
    counts = {}   # start an empty tally of how often each option appeared
    for word in options:   # go through the options one by one
        counts[word] = counts.get(word, 0) + 1   # add one to that word's tally
    words = list(counts.keys())   # the different words it can pick from here
    weights = [counts[word] ** (1 / temperature) for word in words]   # temperature reshapes the odds: a low temperature makes the favourite almost unbeatable, a high one levels every word's chance
    return random.choices(words, weights=weights)[0]   # roll the weighted dice and return the winning word

def write_sentence(temperature, seed):   # define a tool: grow one full sentence out of the prompt
    random.seed(seed)   # fix the dice first: the same seed always produces the same sentence
    words = list(prompt)   # the sentence starts as a copy of the prompt
    while not words[-1].endswith(".") and len(words) < 20:   # keep going until a word carrying a full stop is picked, and never past 20 words
        next_word = pick_next((words[-2], words[-1]), temperature)   # pick the next word from the last two words
        words.append(next_word)   # add the new word to the sentence
    return " ".join(words)   # join the words back into one line of text

# STEP 5 — GENERATE: the same prompt at two temperatures, two personalities.
safe_sentence = write_sentence(0.2, 42)   # temperature 0.2: play it safe — the most learned pattern nearly always wins
print("Temperature 0.2 (safe):", safe_sentence)   # the careful sentence the generator wrote
creative_sentence = write_sentence(1.5, 42)   # temperature 1.5: roll freer — rare words get a real chance
print("Temperature 1.5 (creative):", creative_sentence)   # the wilder sentence from the same prompt

# STEP 6 — A GENERATED PICTURE: points from latent space, decoded into pixels.
random.seed(7)   # fix the dice again, so the picture comes out the same on every run
size = 60   # the picture will be 60 pixels wide and 60 pixels tall
picture = np.zeros((size, size))   # start with a blank black canvas of pixels
stars = 0   # a counter for the background star pixels the decoder lights up
for row in range(size):   # visit every row of the canvas
    for col in range(size):   # visit every pixel in the row
        x = col / size * 2 - 1   # turn the column into a latent number between -1 and 1
        y = 1 - row / size * 2   # turn the row into a latent number between -1 and 1
        distance = (x ** 2 + y ** 2) ** 0.5   # how far this latent point sits from the centre
        if 0.62 < distance < 0.78:   # points inside the ring band decode as the moon ring...
            picture[row, col] = 1.0   # ...so light this pixel fully
        elif random.random() < 0.02:   # any other latent point has a small chance to sparkle...
            picture[row, col] = 0.7   # ...as a faint background star
            stars += 1   # count the new star
plt.imsave("generated_moon.png", picture, cmap="gray")   # save the decoded pixels as a PNG picture file
print("Picture saved as generated_moon.png —", size, "x", size, "pixels,", stars, "background stars")   # report what was generated
