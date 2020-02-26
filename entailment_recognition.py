# import necessary packages
import pandas as pd
import numpy as np
import re
from gensim.models import KeyedVectors
from sklearn.model_selection import StratifiedShuffleSplit
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Flatten, Dropout
from keras.layers import Bidirectional
from keras.optimizers import RMSprop, Adam, SGD, Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt

# load the dataset
training_dataset = pd.read_csv('dataset/train.csv')
testing_dataset = pd.read_csv('dataset/test.csv')

# some helper functions
NUM_OF_WORDS = 100000
def clean_sents(sentence):
    return re.sub('[^A-Za-z\-]+', ' ', str(sentence)).replace("'", '').lower()

def load_w2v_model(model_path, max_vocab_size=100000):
	# pre-download the Google's News Word2Vec from here
	# LINK: https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
    return KeyedVectors.load_word2vec_format(model_path, binary=True, limit=max_vocab_size)

def word_embeddings(sentence, word2id_dict):
    return np.array([word2id_dict[word] if word in word2id_dict else 0 for word in sentence.split(' ')])

# data preparation
train_df = training_dataset.copy()
train_df.sentence1 = train_df.sentence1.apply(clean_sents)
train_df.sentence2 = train_df.sentence2.apply(clean_sents)

# split features from labels
X = train_df.drop(['gold_label'], axis=1)
y = train_df[['gold_label']]

# categories & 1-hot-encode labels
y_encoded, y_categories = y['gold_label'].factorize()
y = to_categorical(y_encoded, num_classes=3)

# load the pre-baked Word2Vec model
w2v_model = load_w2v_model(model_path='GoogleNews-vectors/GoogleNews-vectors-negative300.bin', 
	max_vocab_size=100000)

word2index = {word: i+1 for i, word in enumerate(w2v_model.index2word) if i < NUM_OF_WORDS}

embeddings = w2v_model.vectors[:NUM_OF_WORDS, :]
embeddings = np.concatenate((np.zeros((1,300)), embeddings))
# print(embeddings.shape)

X['x1'] = X.sentence1.apply(lambda x: word_embeddings(x, word2index))
X['x2'] = X.sentence2.apply(lambda x: word_embeddings(x, word2index))

# Stratified Split to create Train, Validation set
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=29)
for train_index, test_index in sss.split(X.drop(['sentence1','sentence2'],axis=1), y):
    X_train = X.drop(['sentence1','sentence2'],axis=1).loc[train_index]
    X_test = X.drop(['sentence1','sentence2'],axis=1).loc[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

# get max sequence length
max_seq_len = max( max(X_train.x1.map(len)), max(X_train.x2.map(len)) )

# add padding till length is about max sequence length
x1_padded = pad_sequences(X_train.x1, maxlen=max_seq_len)
x2_padded = pad_sequences(X_train.x2, maxlen=max_seq_len)
x1_test_padded = pad_sequences(X_test.x1, maxlen=max_seq_len)
x2_test_padded = pad_sequences(X_test.x2, maxlen=max_seq_len)

# concatenate the two sentence embedding arrays, for training
train_set = np.c_[x1_padded, x2_padded]
test_set = np.c_[x1_test_padded, x2_test_padded]

# define keras models
def build_model():
    model = Sequential()
    model.add(Embedding(embeddings.shape[0], embeddings.shape[1], input_length=max_seq_len*2))
    model.add(GRU(units=32, dropout=(0.4), recurrent_dropout=(0.4)))
    model.add(Dropout(rate=0.4))
    model.add(Dense(3, activation='softmax'))
    print(model.summary())

    optimizer = RMSprop(lr=0.001, epsilon=1e-08) #Adam(lr=0.001, epsilon=1e-08)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# setting early stop & model checkpoint
filepath = "data_models/keras-model-epoch-{epoch:02d}-val_acc-{val_acc:.4f}.hdf5"
chk_point = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False)
early_stop = EarlyStopping(monitor='val_acc', patience=3, verbose=1)
callbacks_list = [early_stop, chk_point]

# call build model
keras_model = build_model()

# save history for viewing acc & loss over course of training
history = keras_model.fit(train_set, y_train, batch_size=32, epochs=10, validation_data=(test_set, y_test), 
                          callbacks=callbacks_list)

# view change in acc & loss among training & validation set over epochs
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.xlabel("epochs")
plt.show()

# prep testing dataset
test_df = testing_dataset.copy()
test_df.sentence1 = test_df.sentence1.apply(clean_sents)
test_df.sentence2 = test_df.sentence2.apply(clean_sents)
test_df['x1'] = test_df.sentence1.apply(lambda x: word_embeddings(x, word2index))
test_df['x2'] = test_df.sentence2.apply(lambda x: word_embeddings(x, word2index))
X_new = test_df[['x1','x2']]
x1_padded = pad_sequences(X_new.x1, maxlen=max_seq_len)
x2_padded = pad_sequences(X_new.x2, maxlen=max_seq_len)
X_new_prepared = np.c_[x1_padded, x2_padded]

# make predictions
final_pred = keras_model.predict(X_new_prepared)
final_pred = np.argmax(final_pred, axis=1)
final_df = pd.DataFrame(columns=['gold_label'])
for ix, pred in enumerate(final_pred):
    if pred == 0:
        final_df.loc[len(final_df), 'gold_label'] = 'contradiction'
    elif pred == 1:
        final_df.loc[len(final_df), 'gold_label'] = 'entailment'
    else:
        final_df.loc[len(final_df), 'gold_label'] = 'neutral'
final_df.to_csv('predicted.csv', index=False)