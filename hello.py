from flask import Flask, flash, redirect, render_template, request, session, abort
from random import randint
import numpy as np 
import pandas as pd 
import re
from random import randint
from sklearn.model_selection import train_test_split
import tensorflow as tf

batchSize = 1000
lstmUnits = 64
numClasses = 2
iterations = 100000
maxSeqLength = 200
numDimensions=100

training_epochs = 200
display_step = 1

def loadGloveModel():
    print "Loading Glove Model"
    f = open('glove.6B.100d.txt','r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print "Done.",len(model)," words loaded!"
    return model

# glove word-vec dictionary
glove = loadGloveModel()

glove_vecs = pd.Series(glove)
glove_vecs = np.array(list(glove_vecs.values))
print glove_vecs.shape 

glove_keys=np.array(glove.keys(), dtype=object)
print glove_keys.shape
glove_keys = unicode(glove_keys.tolist())

def word2vec(word):
    try:
        return glove[word]
    except KeyError:
        return np.zeros((100))

def to_binary(X):
    if X == "REAL":
        return 1
    else:
        return 0

def one_hot_encode(labels):
    labels = map(to_binary, labels)
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), [int(l) for l in labels]] = 1
    return one_hot_encode

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [None, numClasses])
#input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.placeholder(tf.float32, shape=[None, maxSeqLength, numDimensions], name='data')
#tf.Variable(tf.zeros([None, maxSeqLength, numDimensions]),dtype=tf.float32)
#data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))




app = Flask(__name__)
 
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        text = request.form["text"]
        #print text
        text_matrix = np.empty((1, maxSeqLength), dtype='object')
        words = np.empty((maxSeqLength), dtype='object')
        indexCounter = 0
        cleanedLine = cleanSentences(text)
        split = cleanedLine.split()
        if len(split)< maxSeqLength:
            indexCounter = maxSeqLength-len(split)
        else:
            split = split[:maxSeqLength]
        for word in split:
            words[indexCounter] = word
            indexCounter = indexCounter + 1
        text_matrix[0] = words
        #print text_matrix
        _data = [word2vec(w) for w in text_matrix[0]]
        _data = np.array(_data).reshape(1,200,100)
        pred = sess.run(prediction, feed_dict={data: _data})
        ind = tf.argmax(pred,1).eval()
        label = ''
        if ind[0] == 0:
            label = 'FAKE'
        else:
            label = "REAL"
        return label
    return render_template("index.html")
 
"""#@app.route("/hello/<string:name>")
@app.route("/hello/<string:name>/")
def hello(name):
#    return name
    quotes = [ "'If people do not believe that mathematics is simple, it is only because they do not realize how complicated life is.' -- John Louis von Neumann ",
               "'Computer science is no more about computers than astronomy is about telescopes' --  Edsger Dijkstra ",
               "'To understand recursion you must first understand recursion..' -- Unknown",
               "'You look at things that are and ask, why? I dream of things that never were and ask, why not?' -- Unknown",
               "'Mathematics is the key and door to the sciences.' -- Galileo Galilei",
               "'Not everyone will understand your journey. Thats fine. Its not their journey to make sense of. Its yours.' -- Unknown"  ]
    randomNumber = randint(0,len(quotes)-1) 
    quote = quotes[randomNumber] 
 
    return render_template(
        'test.html',**locals())"""
 
if __name__ == "__main__":
    app.run()





