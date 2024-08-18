from keras.models import load_model
import random
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import warnings
import pickle
import json
from nltk.stem import WordNetLemmatizer
from flask import Flask, jsonify, request
import nltk
nltk.download('punkt')  # Sentence tokenize
app = Flask(__name__)
lemmatizer = WordNetLemmatizer()
warnings.filterwarnings('ignore')


# Chargement des données depuis le fichier .pkl
with open('words.pkl', 'rb') as fichier:
    contenu = pickle.load(fichier)

# Vérification du type de contenu chargé
if isinstance(contenu, list):
    words = contenu
    print(words)
else:
    print("Le contenu du fichier .pkl n'est pas une liste.")

with open('classes.pkl', 'rb') as fichier:
    contenu = pickle.load(fichier)

# Vérification du type de contenu chargé
if isinstance(contenu, list):
    classes = contenu
    print(classes)
else:
    print("Le contenu du fichier .pkl n'est pas une liste.")


# Load the model from the .h5 file
model = load_model('chatbot_model.h5')

intents = json.loads(open('intents1.json').read())  # load json file


def clean_up_sentence(sentence):

    # tokenize the pattern - split words into array

    sentence_words = nltk.word_tokenize(sentence)
    # print(sentence_words)
    # stem each word - create short form for word

    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    # print(sentence_words)

    return sentence_words


def bow(sentence, words, show_details=True):

    # tokenize the pattern
    # clean_up_sentence("hello amine how jhjjh are you ?") retourne=>['hello', 'amine', 'how', 'jhjjh', 'are', 'you', '?']
    sentence_words = clean_up_sentence(sentence)
    # print(sentence_words)

    # bag of words - matrix of N words, vocabulary matrix

    bag = [0]*len(words)
    # print(bag)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
                #print ("found in bag: %s" % w)
    # print(bag)
    return (np.array(bag))


def predict_class(sentence, model):

    # filter out predictions below a threshold

    p = bow(sentence, words, show_details=False)
    # print(p)

    res = model.predict(np.array([p]))[0]
    print("res", len(res))

    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # print(results)
    # sort by strength of probability

    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)

    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list


def getResponse(ints, intents_json):

    tag = ints[0]['intent']
    # print(tag)

    list_of_intents = intents_json['intents']
    # print(list_of_intents)

    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(text):
    ints = predict_class(text, model)
    # print(ints)
    res = getResponse(ints, intents)
    return res


@app.route('/api/hello', methods=['GET'])
def hello():
    # Get the 'query' parameter from the URL
    query = request.args.get('query', '')
    if query in ['quit', 'exit', 'bye']:
        return "Goodbye!"  # Exit message
    try:
        # Assuming you have a function chatbot_response
        res = chatbot_response(query)
        return jsonify({"response": res})
    except:
        return jsonify({"response": "You may need to rephrase your question."})


if __name__ == '__main__':
    app.run(host='192.168.1.18', port=5000)
