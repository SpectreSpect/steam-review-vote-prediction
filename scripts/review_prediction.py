import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
import pickle
import numpy as np


def preprocess_train_data(train: pd.DataFrame):
    train.dropna(subset=['review'], inplace=True)
    train = train.drop_duplicates(['review'])
    return train


def create_and_fit_tokenizer(data, max_words_num):
    # num_words = 20000
    # total_vocabulary_size = max_words_num + 2 # to take into account start and end tokens
    tokenizer = Tokenizer(num_words=max_words_num, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True,
                        split=' ', char_level=False)
    tokenizer.fit_on_texts(data['review'])
    return tokenizer


def get_tokenizer(pickle_file_path) -> Tokenizer:
    with open(pickle_file_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def predict_reaction(model, tokenizer, input_text):
    input_data = tokenizer.texts_to_sequences([input_text])
    predictions = model.predict(input_data)
    return np.argmax(predictions)




model = tf.keras.models.load_model("../models/1")
model.summary()

# train = pd.read_csv("../data/reviews/98/reviews.csv", nrows=1000)
# train = preprocess_train_data(train)
# max_words_num = 20000
# checkpoint_name = 'checkpoint.model.keras'


# show_voted_up_distribution(train)


# tokenizer = create_and_fit_tokenizer(train, max_words_num - 2)
# tokens_count = min(max_words_num, len(tokenizer.index_word)) + 1

tokenizer = get_tokenizer("../models/tokenizer/tokenizer.pickle")

# with open("../models/tokenizer/tokenizer.pickle", 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("1:", predict_reaction(model, tokenizer, "It's Overwatch. Appreciate it being added to steam."))



# input_data = ["It's Overwatch. Appreciate it being added to steam."]

# print(tokenizer.index_word[4])
# print(tokenizer.index_word[80])


# input_data = tokenizer.texts_to_sequences(input_data)
# print(input_data)

# print(model.predict(input_data))
# print(model.predict([[456, 234, 634, 23]]))
# print(model.predict([[456, 234, 634, 23]]))
# print(model.predict([[100, 100, 150, 234]]))
# print(model.predict([[654, 92, 34, 12]]))

# predictions = model.predict(input_data)
# print(predictions)

