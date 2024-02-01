import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import os
import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

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


def get_review_sequences_and_labels(data, tokenizer, batch_size):
    review_sequences = tokenizer.texts_to_sequences(data['review'])
    max_seq_len = max([len(s) for s in review_sequences])
    review_sequences = pad_sequences(review_sequences, padding='post')

    labels = to_categorical(data['voted_up'], num_classes=2)

    dataset = tf.data.Dataset.from_tensor_slices((review_sequences, labels))
    dataset = dataset.shuffle(buffer_size=len(data))

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    validation_size = int(len(train) * 0.0)

    train_dataset = dataset.skip(validation_size).batch(batch_size).cache()
    validation_dataset = dataset.take(validation_size).batch(batch_size).cache()

    return train_dataset, validation_dataset


def create_and_fit_tokenizer(data, max_words_num):
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

train = pd.read_csv("../data/reviews/98/reviews.csv", nrows=1000)
train = preprocess_train_data(train)
max_words_num = 20000
checkpoint_name = 'checkpoint.model.keras'


# show_voted_up_distribution(train)


tokenizer = create_and_fit_tokenizer(train, max_words_num - 2)

tokens_count = min(max_words_num, len(tokenizer.index_word)) + 1
# review_sequences, labels = get_review_sequences_and_labels(train, tokenizer)
train_dataset, validation_dataset = get_review_sequences_and_labels(train, tokenizer, 128)

loaded_model = tf.keras.models.load_model('../loaded_models_for_test/1')
loaded_model.evaluate(train_dataset)
exit()


checkpoint_filepath = checkpoint_dir + '/' + checkpoint_name
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max')

log_filepath = checkpoint_dir + '/' + 'log.csv'
csv_logger = tf.keras.callbacks.CSVLogger(log_filepath, append=True)

if flag == 0:   # start training
    model = build_model(tokens_count, 2)
elif flag == 1:
    model = tf.keras.models.load_model(checkpoint_filepath)

model.summary()
# exit()

# print(train_dataset.element_spec)

model.fit(train_dataset,
          validation_data=validation_dataset,
          epochs=50,
          callbacks=[model_checkpoint_callback, csv_logger])
