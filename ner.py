import pandas as pd
import numpy as np
import math
import numpy
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras import optimizers

dir_path = "C:\\Users\\Tadeas\\Downloads\\ner\\ner_dataset.csv"
everything = pd.read_csv(dir_path, encoding="latin-1")

print(everything)

tags = []

train_batch = 15000


for i in everything["Tag"]:
  i = i.split("B-")
  i = ("").join(i)
  if "I" not in i:
    if i not in tags:
      tags.append(i)

  else:
    pass

print(tags)

def TF_IDF(text):
  words = []
  for i in text:
    i = i.lower()
    for x in i.split(" "):
      if x not in words:
        words.append(x)
 
  k = []
  for i in text:
    i = i.lower()
    for x in i.split(" "):
      k.append(x)
  
  tf = dict()
  for i in words:
    c = 0
    for x in k:
      if x == i:
        c += 1
      
      else:
        pass
    
    num = c / len(k)

    tf[i] = num
  
  
  idf = dict()
  for i in words:
    q = 0
    for x in text:
      x = x.lower()
      if i in x:
        q += 1 

      else:
        pass
  
    b = len(text) / q

    idf[i] = math.log2(b)

  tfidf = dict()

  for i in words:
    fin_num = tf[i] * idf[i]
    tfidf[i] = fin_num

  matrix = []
  for i in range(len(text)):
    matrix.append([])

  for x in matrix:
    for y in range(len(words)):
      x.append(0)
  
  for r, i in enumerate(text):
    i = i.lower()
    for x in i.split(" "):
      u = list(tfidf).index(x)
      matrix[r][u] = tfidf[x]


  return matrix


words = []

for i in everything["Word"][0:20000]:
  words.append(i)

vectorcs = TF_IDF(words)

vectorcs = np.asarray(vectorcs)
vectorcs = vectorcs.reshape(20000, vectorcs.shape[1], 1)

print(vectorcs.shape)


labels = []


for m in everything["Tag"][0:20000]:
  for x, i in enumerate(tags):
    if "B" in m:
      q = m.split("B-")
      if q[1] == i:
        labels.append(x)

      else:
        pass

    elif "I" in m:
      q = m.split("I-")
      
      if q[1] == i:
        labels.append(x)
      
      else:
        pass

    else:
      if m == i:
        labels.append(x)
  

labels = to_categorical(labels, num_classes=len(tags))

realsamples = np.arange(vectorcs.shape[0])
np.random.shuffle(realsamples)
vectorcs = vectorcs[realsamples]
labels = labels[realsamples]

x_train = vectorcs[:train_batch]
x_val = vectorcs[train_batch:]
y_train = labels[:train_batch]
y_val = labels[train_batch:]


m = models.Sequential()

m.add(layers.LSTM(128, input_shape=(4057, 1)))
m.add(layers.Dense(len(tags), activation="softmax"))

m.compile(optimizer=optimizers.Adam(lr=0.001), loss="categorical_crossentropy", metrics= ["acc"])

m.fit(x_train, y_train, epochs= 30, batch_size= 64, validation_data=(x_val, y_val))

m.save("Ner.h5")


  
  

  


