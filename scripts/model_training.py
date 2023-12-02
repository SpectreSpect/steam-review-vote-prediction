import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

file_path = "../data/reviews/14/reviews.csv"

print("Start data loading.")
train = pd.read_csv(file_path)
print("The train data loading has been completed.\n")

print(f"Columns: {train.columns}")
print("Train dataframe info: \n")
print(train.info())

# The review column has some missing values. It's better to just delete them because these are few.
print("\nDropping rows with missing values.")
train.dropna(subset=['review'], inplace=True)
print("Dropping has been completed. Resulted array: \n")
print(train.info())

unique_voted_up_values = train['voted_up'].astype("string").unique()
print(unique_voted_up_values)
counts = train['voted_up'].value_counts().to_numpy()
print(counts)

fig, ax = plt.subplots()

bar_labels = ['red', 'blue']
bar_colors = ['tab:red', 'tab:blue']

rects = ax.bar(unique_voted_up_values, counts, label=unique_voted_up_values, color=bar_colors)

ax.set_ylim([0, np.max(counts) + np.max(counts) * 0.2])
ax.bar_label(rects, padding=3)
ax.set_title('Target count in training set')

plt.show()

# There are many more true values than false ones, which is bad.

num_words = 20000
total_vocabulary_size = num_words + 2 # to take into account start and end tokens
tokenizer = Tokenizer(num_words=num_words, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True,
                      split=' ', char_level=False)
tokenizer.fit_on_texts(train['review'])
review_sequences = tokenizer.texts_to_sequences(train['review'])
tokens_count = min(num_words, len(tokenizer.index_word))
start_token = tokens_count
end_token = tokens_count + 1
for i in range(len(review_sequences)):
    review_sequences[i] = [start_token] + review_sequences[i] + [end_token]

max_seq_len = max([len(s) for s in review_sequences])
review_sequences = tf.constant([s + [0] * (max_seq_len - len(s)) for s in review_sequences])

review_sequences = tf.convert_to_tensor(review_sequences)
num_label_classes = train['voted_up'].nunique()
labels = to_categorical(train['voted_up'], num_classes=num_label_classes)
labels = tf.convert_to_tensor(labels)

inputs = tf.keras.layers.Input(shape=1)
embedding_outputs = tf.keras.layers.Embedding(input_dim=total_vocabulary_size, output_dim=10, input_shape=(None,))(inputs)
lstm_outputs_1 = tf.keras.layers.LSTM(10, return_sequences=True)(embedding_outputs)
lstm_outputs_2 = tf.keras.layers.LSTM(10, return_sequences=True)(lstm_outputs_1)
lstm_outputs_3 = tf.keras.layers.LSTM(10)(lstm_outputs_2)
outputs = tf.keras.layers.Dense(num_label_classes, 'softmax')(lstm_outputs_3)

model = tf.keras.Model(inputs, outputs)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(review_sequences, labels, epochs=10, batch_size=128, validation_split=0.2)
