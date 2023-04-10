from flask import Flask, render_template, request
import tflearn
from time import sleep
import pickle
import json
import random
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from nltk.stem.lancaster import LancasterStemmer
import nltk
nltk.download('punkt')
stemmer = LancasterStemmer()


with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Define the TensorFlow graph
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Define the model and the training parameters
model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)

# Save the trained model
model.save("model.tflearn")

# Test the model with some example input


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def process(message):
    inp = message
    results = model.predict([bag_of_words(inp, words)])[0]
    results_index = np.argmax(results)
    tag = labels[results_index]
    if results[results_index] > 0.8:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        sleep(1)
        Bot = random.choice(responses)
        return(Bot)
    else:
        return("I don't understand!")


app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_reponse():
    userText = request.args.get('msg')
    return str(process(userText))


if __name__ == "__main__":
    app.run(debug=True)
