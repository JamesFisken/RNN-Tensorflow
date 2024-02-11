import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, SpatialDropout1D, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import re
import ast


'''
To do:
-Somehow use the GPU in place of the CPU
-Adjust loss function to recognize words of similar semantic value
-Find another larger dataset
-Test various parameters
'''


#change number of threads as you see fit
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

max_sequence_length = 64  # Each training step consists of this amount of tokens
vocab_size = 30000
training = False
train_loaded_NN = False
loaded_NN = "MovieRNN.keras"  # file name of model to load
Saved_NN = "MovieRNN.keras"  # file name of model to save

dialogue_spliter = 'ඞ'
oov_token = 'xxxxxxx'

def separate_punctuation(text):
    # Use regular expression to separate punctuation from words
    separated_text = re.sub(r'(\w+)([."—:,!?‘-])', r'\1 \2 ', text)
    separated_text = re.sub(r'([."—:,!?‘-])(\w+)', r' \1 \2', separated_text)

    # Separate contractions
    separated_text = re.findall(r"'\w+|n't|\w+(?=n't)|\w+|[^\s\w]", separated_text)

    separated_text = " ".join(separated_text)
    return separated_text


def split_sting(input_string, part_length):
    words = input_string.split()
    return [' '.join(words[i:i + part_length]) for i in range(0, len(words), part_length)]

def filter_to_length(sequence, length):
    filtered_sequence = []
    for line in sequence:
        if len(line.split()) >= length:  # if the length of the line is adequate append them to the resultant list
            filtered_sequence.append(line)
    return filtered_sequence

line_dict = {}
dialogs = []
sequences = []

with open('cornell movie-dialogs corpus/movie_lines.txt', 'r') as f:
    lines = f.read()

for line in lines.split('\n'):
    if line.strip():
        parts = line.split(' +++$+++ ')  # split based on categories seperated by ' +++$+++ '
        line_dict[parts[0]] = parts[4]  # add dict value key pair of the line code and corresponding dialog

with open('cornell movie-dialogs corpus/movie_conversations.txt', 'r') as f:
    lines = f.read()
for line in lines.split('\n'):
    decoded_sequence = ""
    if line.strip():
        coded_sequence = line.split(' +++$+++ ')[3]
        coded_sequence = ast.literal_eval(coded_sequence)
        for part in coded_sequence:
            decoded_sequence = decoded_sequence + separate_punctuation(line_dict[part]) + f' {dialogue_spliter} '  # add strings

        sequences.append(separate_punctuation(decoded_sequence))
#sequences = filter_to_length(sequences, 50)  # optional - removes sequences that are too short

'''
daily_dialog_data = pd.read_csv('dailydialog.txt', sep='\t', header=None)   # upload training data
dialogs = daily_dialog_data[0].tolist()  # convert csv file type to a list

#preprocessing
dialogs = [text.replace('__eou__', '') for text in dialogs]  # remove eou tag
dialogs = ''.join(dialogs)
dialogs = separate_punctuation(dialogs)  # changes scenarios like 'how dare you!'  to 'how dare you !' for tokenization


sequences = split_sting(dialogs, max_sequence_length)
'''

#print(sequences[0])


# Tokenization and Padding

tokenizer = Tokenizer(filters='*#$%&<>@[\\]^_`{}~\t\n',num_words=vocab_size, oov_token=oov_token)  # Includes only useful punctuation: ,.'":;+-=()
tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences)  # tokenizes words



word_index = tokenizer.word_index
print(word_index)
print(len(sequences))


if training:
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')  # Adds 0s to ensure uniformity

    # Generate input and target sequences for predicting the next word offset by one word
    input_sequences = padded_sequences[:, :-1]
    target_sequences = padded_sequences[:, 1:]

    # Split the dataset into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(input_sequences, target_sequences, test_size=0.1, random_state=41)

    # Modify amount of data given to the model
    print(len(input_sequences))
    X_train = input_sequences[0:1000]
    y_train = target_sequences[0:1000]

    if not train_loaded_NN:
        # Define the model
        embedding_dim = 200

        model = Sequential([
            Embedding(input_dim=vocab_size+1, output_dim=embedding_dim, input_length=max_sequence_length - 1, mask_zero=True),
            SpatialDropout1D(0.2),
            Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
            BatchNormalization(),
            Dense(vocab_size, activation='softmax')  # Tested: Sigmoid, Relu, Softmax
        ])

        # Compile the model
        optimizer = keras.optimizers.Adamax(learning_rate=0.003, beta_1=0.9, beta_2=0.999)  # Parameters need tuning
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    else:
        model = tf.keras.models.load_model(loaded_NN)

    # Print a summary of the model
    model.summary()

    # Train the model
    model.fit(X_train, np.expand_dims(y_train, -1), epochs=5, batch_size=32)

    model.save(Saved_NN)

else:
    model = tf.keras.models.load_model(loaded_NN)
    model.summary()

while True:
    input_text = input("What is your question: ")

    for _ in range(3):  # Predict next 10 words
        input_sequence = pad_sequences([tokenizer.texts_to_sequences([separate_punctuation(input_text)])[0]], maxlen=max_sequence_length - 1, padding='post')[0]
        actual_text = tokenizer.sequences_to_texts([input_sequence])[0]
        end_of_sentence = len(actual_text.split())


        total_predictions = model.predict(np.array([input_sequence]))


        next_word_predictions = total_predictions[0][end_of_sentence - 1]

        # Gets the top 5 predicted words for future variability
        top_predictions_index = np.argsort(next_word_predictions)[-5:][::-1]

        index_value_dict = {index: next_word_predictions[index] for index in top_predictions_index}
        #print(index_value_dict)

        # takes the 2nd prediction in the case that the first is a blank space
        print(top_predictions_index[0], '')
        if top_predictions_index[0] == 0 or top_predictions_index[0] == 3:
            print(top_predictions_index[1], " here")
            prediction_index = top_predictions_index[1]
        else:
            prediction_index = top_predictions_index[0]
        #predicted_word_index = np.argmax(predictions[0][end_of_sentence - 1])

        predicted_word = tokenizer.index_word.get(prediction_index, '')

        input_text = ' '.join(actual_text.split()) + ' ' + predicted_word

    # Print the results
    print(input_text)
