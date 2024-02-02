import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pickle


def get_tokenizer(pickle_file_path: str) -> Tokenizer:
        with open(pickle_file_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer


max_words_num = 20000
tokenizer = get_tokenizer("../models/tokenizer/tokenizer.pickle")

model = tf.keras.models.load_model("../models/1")
print("The input loop has started and you can now enter prompts.")
while True:
    prompt = input()
    sequences = tokenizer.texts_to_sequences([prompt])
    predictions = model.predict(sequences, verbose=0)
    print(f"positive: {predictions[0][1]}   negative: {predictions[0][0]}")