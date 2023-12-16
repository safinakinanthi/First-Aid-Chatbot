import json
import nltk
import numpy as np
import random
import tensorflow as tf
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()


from nltk.tokenize import word_tokenize

# Load your pre-trained model (replace 'your_model.h5' with your actual model file)
model = tf.keras.models.load_model('EngChatbotModel.h5')

# Load your data from JSON (replace 'your_data.json' with your actual JSON file)
with open('EnglishData.json', 'r') as file:
    data = json.load(file)

# Extract words and labels
words = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds=nltk.word_tokenize(pattern)
        words.extend(wrds)
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
# Word Steamming
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)
# Initialize NLTK stemmer
stemmer = LancasterStemmer()

# Function to convert input to a bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return [bag]

# Function to get a response from the chatbot
def get_response(inp):
    results = model.predict(np.array(bag_of_words(inp, words)))
    results_index = np.argmax(results)
    tag = labels[results_index]

    for intent in data['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']

    return random.choice(responses)

# Main chat loop
print("Start talking with the bot (type quit to stop)!")
while True:
    inp = input("You: ")
    if inp.lower() == "quit":
        break

    response = get_response(inp)
    print(response)
