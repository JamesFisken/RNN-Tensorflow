import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, SpatialDropout1D, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#change number of threads as you see fit
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)


def split_sting(input_string, part_length):
    words = input_string.split()
    return [' '.join(words[i:i + part_length]) for i in range(0, len(words), part_length)]


daily_dialog_data = pd.read_csv('dailydialog.txt', sep='\t', header=None)   # upload training data
dialogs = daily_dialog_data[0].tolist()


dialogs = [text.replace('__eou__', '') for text in dialogs]  # remove eou tag
dialogs = ''.join(dialogs)


max_sequence_length = 64  # Each training step consists of this amount of tokens
sequences = split_sting(dialogs, max_sequence_length)

# Tokenization and Padding
tokenizer = Tokenizer(filters='#$%&<>@[\\]^_`{|}~\t\n',)  # Includes only useful punctuation: ,.'":;+-=()
tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences) # tokenizes words

word_index = tokenizer.word_index
print(word_index)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')  # Adds 0s to ensure uniformity

# Generate input and target sequences for predicting the next word offset by one word
input_sequences = padded_sequences[:, :-1]
target_sequences = padded_sequences[:, 1:]


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_sequences, target_sequences, test_size=0.1, random_state=41)

'''
repetitions = 500
X_train = np.tile(input_sequences[0:3], (repetitions, 1))
y_train = np.tile(target_sequences[0:3], (repetitions, 1))
'''

print(len(X_train))
# Modify amount of data given to the model
X_train = X_train[0:1000]
y_train = y_train[0:1000]


# Define the model
embedding_dim = 200
vocab_size = len(word_index) + 1

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length - 1, mask_zero=True),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)),
    BatchNormalization(),
    Dense(vocab_size, activation='softmax')  # Tested: Sigmoid, Relu, Softmax
])

# Compile the model
optimizer = keras.optimizers.Adamax(learning_rate=0.003, beta_1=0.9, beta_2=0.999)  # Parameters need tuning
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model
model.summary()

# Train the model
model.fit(X_train, np.expand_dims(y_train, -1), epochs=10, batch_size=32)

while True:
    input_text = input("What is your question: ")

    for _ in range(5):  # Predict next 5 words
        input_sequence = pad_sequences([tokenizer.texts_to_sequences([input_text])[0]], maxlen=max_sequence_length - 1, padding='post')[0]
        actual_text = tokenizer.sequences_to_texts([input_sequence])[0]
        end_of_sentence = len(actual_text.split())


        total_predictions = model.predict(np.array([input_sequence]))


        next_word_predictions = total_predictions[0][end_of_sentence - 1]

        # Gets the top 10 predicted words for future variability
        top_predictions_index = np.argsort(next_word_predictions)[-5:][::-1]

        index_value_dict = {index: next_word_predictions[index] for index in top_predictions_index}
        print(index_value_dict)

        # takes the 2nd prediction in the case that the first is a blank space
        if top_predictions_index[0] == 0:
            prediction_index = top_predictions_index[1]
        else:
            prediction_index = top_predictions_index[0]
        #predicted_word_index = np.argmax(predictions[0][end_of_sentence - 1])

        predicted_word = tokenizer.index_word.get(prediction_index, '')

        input_text = ' '.join(actual_text.split()) + ' ' + predicted_word

    # Print the results
    print(input_text)
