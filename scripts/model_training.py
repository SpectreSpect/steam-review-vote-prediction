import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import os
import re
import pickle


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


def generate_output_dir(outdir, run_desc):
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(\
            os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(run_dir)
    os.makedirs(run_dir)
    return run_dir


def preprocess_train_data(train: pd.DataFrame):
    train.dropna(subset=['review'], inplace=True)
    train = train.drop_duplicates(['review'])
    return train


def show_voted_up_distribution(data):
    unique_voted_up_values = data['voted_up'].astype("string").unique()
    counts = data['voted_up'].value_counts().to_numpy()

    fig, ax = plt.subplots()

    bar_colors = ['tab:red', 'tab:blue']

    rects = ax.bar(unique_voted_up_values, counts, label=unique_voted_up_values, color=bar_colors)

    ax.set_ylim([0, np.max(counts) + np.max(counts) * 0.2])
    ax.bar_label(rects, padding=3)
    ax.set_title('Target count in training set')
    plt.show()


def get_review_sequences_and_labels(data, tokenizer):
    review_sequences = tokenizer.texts_to_sequences(data['review'])
    # tokens_count = min(max_words_num, len(tokenizer.index_word))
    # start_token = tokens_count
    # end_token = tokens_count + 1
    max_seq_len = max([len(s) for s in review_sequences])
    review_sequences = tf.constant([s + [0] * (max_seq_len - len(s)) for s in review_sequences])

    review_sequences = tf.convert_to_tensor(review_sequences)
    labels = to_categorical(data['voted_up'], num_classes=2)
    labels = tf.convert_to_tensor(labels)
    return review_sequences, labels


def create_and_fit_tokenizer(data, max_words_num):
    # num_words = 20000
    # total_vocabulary_size = max_words_num + 2 # to take into account start and end tokens
    tokenizer = Tokenizer(num_words=max_words_num, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True,
                        split=' ', char_level=False)
    tokenizer.fit_on_texts(data['review'])
    return tokenizer


def build_model(vocab_size, num_classes):
    inputs = tf.keras.layers.Input(shape=1)
    embedding_outputs = tf.keras.layers.Embedding(input_dim=vocab_size, 
                                                  output_dim=10, 
                                                  input_shape=(None,),
                                                  mask_zero=True)(inputs)
    lstm_outputs_1 = tf.keras.layers.LSTM(10, return_sequences=True)(embedding_outputs)
    lstm_outputs_2 = tf.keras.layers.LSTM(10, return_sequences=True)(lstm_outputs_1)
    lstm_outputs_3 = tf.keras.layers.LSTM(10)(lstm_outputs_2)
    outputs = tf.keras.layers.Dense(num_classes, 'softmax')(lstm_outputs_3)

    model = tf.keras.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics='accuracy')
    return model


flag = int(input("What do you want to do?(0 - start training, 1 - resume training): "))

if flag == 0:
    checkpoint_dir = generate_output_dir('../models/', "test-train")
elif flag == 1:
    checkpoint_dir = "../models/00001-test-train/"

train = pd.read_csv("../data/reviews/1/reviews.csv")
train = preprocess_train_data(train)
max_words_num = 20000
checkpoint_name = 'checkpoint.model.keras'

# show_voted_up_distribution(train)


tokenizer = create_and_fit_tokenizer(train, max_words_num - 2)

tokens_count = min(max_words_num, len(tokenizer.index_word)) + 1
review_sequences, labels = get_review_sequences_and_labels(train, tokenizer)

checkpoint_filepath = checkpoint_dir + '/' + checkpoint_name
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max')

log_filepath = checkpoint_dir + '/' + 'log.csv'
csv_logger = tf.keras.callbacks.CSVLogger(log_filepath, append=True)

if flag == 0: # start training
    model = build_model(tokens_count, 2)
elif flag == 1:
    model = tf.keras.models.load_model(checkpoint_filepath)

model.summary()
# exit()

model.fit(review_sequences, labels, 
          epochs=50, 
          batch_size=128, 
          validation_split=0.2,
          callbacks=[model_checkpoint_callback, csv_logger])


