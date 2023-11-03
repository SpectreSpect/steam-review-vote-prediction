import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import keras.layers
import numpy as np

file_path = "../data/reviews/cat.csv"

num_classes = 2
count = 0
chunk_size = 1000

csv_reader = pd.read_csv(file_path, chunksize=chunk_size, iterator=True)

# for chunk in csv_reader:
#     print(count)
#     count += len(chunk)
num_words = 20000
tokenizer = Tokenizer(num_words=num_words, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True, split=' ', char_level=False)

i = 0
for chunk in csv_reader:
    if i > 100:
        break
    tokenizer.fit_on_texts(chunk['review'].astype(str))
    count += len(chunk)
    if i % 100 == 0:
        print(f'chunk: {count}     words count: {len(tokenizer.word_index)}')
    i += 1

print(len(tokenizer.word_index))
print(count)




inputs = keras.layers.Input(1)
embeddings = keras.layers.Embedding(input_dim=num_words, output_dim=256, input_shape=(None,))(inputs)
lstm_1 = keras.layers.LSTM(256, return_sequences=True)(embeddings)
lstm_2 = keras.layers.LSTM(256, return_sequences=True)(lstm_1)
lstm_3 = keras.layers.LSTM(256)(lstm_2)
outputs = keras.layers.Dense(num_classes, activation='softmax')(lstm_3)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])


for chunk in csv_reader:
    reviews = chunk['review'].astype(str)
    votes = to_categorical(chunk['voted_up'], num_classes=num_classes)
    
    review_sequences = tokenizer.texts_to_sequences(reviews)
    
    max_len = max([len(s) for s in review_sequences])
    review_sequences = tf.constant([s + [0] * (max_len - len(s)) for s in review_sequences])
    
    model.fit(review_sequences, votes, batch_size=32, epochs=10)