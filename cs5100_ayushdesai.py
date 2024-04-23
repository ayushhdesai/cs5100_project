import numpy as np
import pandas as pd
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, TimeDistributed, Concatenate, RepeatVector, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings, re
warnings.filterwarnings('ignore')
import tensorflow as tf
from tqdm.notebook import tqdm
tqdm.pandas()
from tensorflow.keras import backend as K
logger = tf.get_logger()
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

ai_df = pd.read_csv('deu.txt',delimiter='\t',header=None)
ai_df.columns = ['English','German','LangSource']
ai_df.head()

ai_df.drop('LangSource',axis=1,inplace=True)

ai_df.shape

ai_df.isna().sum()

ai_df.duplicated().sum()

ai_df['eng_len'] = ai_df.English.apply(len)
ai_df['ger_len'] = ai_df.German.apply(len)

top_e_len = 40
top_g_len = 40

def cleaning_procedure(t):
    t = t.lower()
    p = re.compile('\W')
    t = re.sub(p,' ',t).strip()
    return t

ai_df.English = ai_df.English.progress_apply(cleaning_procedure)
ai_df.German = ai_df.German.progress_apply(cleaning_procedure)

ai_df.German = ai_df.German.apply(lambda x: '<START> ' + x + ' <END>')

english_token = Tokenizer()
english_token.fit_on_texts(ai_df.English)

english_voc_size = len(english_token.word_index) + 1
english_voc_size
print("English Vocab Size:", english_voc_size)

english_seq = english_token.texts_to_sequences(ai_df.English)
english_pseq = pad_sequences(english_seq,maxlen=top_e_len,dtype='int32',padding='post',truncating='post')

german_token = Tokenizer()
german_token.fit_on_texts(ai_df.German)

german_voc_size = len(german_token.word_index) + 1
print("German Vocab Size:", german_voc_size)

german_seq = german_token.texts_to_sequences(ai_df.German)
german_pseq = pad_sequences(german_seq,maxlen=top_g_len,dtype='int32',padding='post',truncating='post')

class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs

        logger.debug(f"encoder_out_seq.shape = {encoder_out_seq.shape}")
        logger.debug(f"decoder_out_seq.shape = {decoder_out_seq.shape}")

        def energy_step(inputs, states):

            logger.debug("Running energy computation step")

            if not isinstance(states, (list, tuple)):
                raise TypeError(f"States must be an iterable. Got {states} of type {type(states)}")

            encoder_full_seq = states[-1]

            W_a_dot_s = K.dot(encoder_full_seq, self.W_a)

            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)

            logger.debug(f"U_a_dot_h.shape = {U_a_dot_h.shape}")

            Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)

            logger.debug(f"Ws_plus_Uh.shape = {Ws_plus_Uh.shape}")

            e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)

            e_i = K.softmax(e_i)

            logger.debug(f"ei.shape = {e_i.shape}")

            return e_i, [e_i]

        def context_step(inputs, states):

            logger.debug("Running attention vector computation step")

            if not isinstance(states, (list, tuple)):
                raise TypeError(f"States must be an iterable. Got {states} of type {type(states)}")

            encoder_full_seq = states[-1]

            c_i = K.sum(encoder_full_seq * K.expand_dims(inputs, -1), axis=1)

            logger.debug(f"ci.shape = {c_i.shape}")

            return c_i, [c_i]

        fake_state_c = K.sum(encoder_out_seq, axis=1)
        fake_state_e = K.sum(encoder_out_seq, axis=2)

        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e], constants=[encoder_out_seq]
        )

        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c], constants=[encoder_out_seq]
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]

# Clearing the session
tf.keras.backend.clear_session()

# Parameters
latent_dim = 256
embedding_dim = 100

# Function to create a new model
def build_model(optimizer):
    # Encoder
    encoder_inputs = Input(shape=(top_e_len,))
    encoder_emb = Embedding(top_g_len, embedding_dim, trainable=True)(encoder_inputs)
    encoder_lstm, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_emb)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_emb = Embedding(german_voc_size, embedding_dim, trainable=True)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=encoder_states)
    decoder_dense = Dense(german_voc_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Defining the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics='accuracy')

    return model

# Building and training the first model
model_adam = build_model('adam')
X_train, X_test, y_train, y_test = train_test_split(english_pseq, german_pseq, test_size=0.25, shuffle=True, random_state=101)
r = model_adam.fit([X_train, y_train[:, :-1]], y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:, 1:], epochs=8, batch_size=256, validation_data=([X_test, y_test[:, :-1]], y_test.reshape(y_test.shape[0], y_test.shape[1], 1)[:, 1:]))

# Clearing session again to avoid any potential overlap
tf.keras.backend.clear_session()

# Building and training the second model
model_rmsprop = build_model('rmsprop')
X_train1, X_test1, y_train1, y_test1 = train_test_split(english_pseq, german_pseq, test_size=0.25, shuffle=True, random_state=101)
s = model_rmsprop.fit([X_train1, y_train1[:, :-1]], y_train1.reshape(y_train1.shape[0], y_train1.shape[1], 1)[:, 1:], epochs=8, batch_size=256, validation_data=([X_test1, y_test1[:, :-1]], y_test1.reshape(y_test1.shape[0], y_test1.shape[1], 1)[:, 1:]))

# Clearing session again to avoid any potential overlap
tf.keras.backend.clear_session()

# Buildind and training the third model
model_lr = build_model('adam')
X_train2, X_test2, y_train2, y_test2 = train_test_split(english_pseq, german_pseq, test_size=0.25, shuffle=True, random_state=101)
lr = model_lr.fit([X_train2, y_train2[:, :-1]], y_train2.reshape(y_train2.shape[0], y_train2.shape[1], 1)[:, 1:], epochs=8, batch_size=256, validation_data=([X_test2, y_test2[:, :-1]], y_test2.reshape(y_test1.shape[0], y_test2.shape[1], 1)[:, 1:]))

model_adam.save('model_adam.h5')
model_rmsprop.save('model_rmsprop.h5')
model_lr.save('model_lr.h5')

plt.plot(r.history['loss'],'r',label='Adam')
plt.plot(s.history['loss'],'b',label='Rmsprop')
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.title('Loss Comparison for Optimizers')
plt.legend()

plt.plot(r.history['accuracy'],'r',label='Adam')
plt.plot(s.history['accuracy'],'b',label='Rmsprop')
plt.xlabel('No. of Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy  Comparison for Optimizers')
plt.legend()

plt.plot(r.history['loss'],'r',label='0.001')
plt.plot(lr.history['loss'],'b',label='0.01')
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.title('Loss Comparison for Different Learning rates')
plt.legend()

plt.plot(r.history['accuracy'],'r',label='0.001')
plt.plot(lr.history['accuracy'],'b',label='0.01')
plt.xlabel('No. of Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy  Comparison for Different Learning Rates')
plt.legend()

from nltk.translate.bleu_score import corpus_bleu

def evaluate_bleu_score(model, X_test, y_test, encoder_model, decoder_model, german_token):
    predictions, references = [], []
    for input_seq, target_seq in zip(X_test, y_test):
        # Decoding the sequence
        decoded_sentence = decode_sequences(input_seq.reshape(1, -1), model, encoder_model, decoder_model, german_token)
        # Split decoded sentence into words, removing the tokens
        decoded_words = decoded_sentence.split()
        if '<END>' in decoded_words:
            decoded_words.remove('<END>')  # Optional: remove end token if present
        predictions.append(decoded_words)
        # Assuming target sequences are already in the correct format
        references.append([target_seq.split()])

    # Calculate BLEU score
    bleu_score = corpus_bleu(references, predictions)
    return bleu_score

# Example of evaluating BLEU score for one model
bleu_score_adam = evaluate_bleu_score(model_adam, X_test, y_test, encoder_model, decoder_model, german_token)
print("BLEU Score for Model Adam:", bleu_score_adam)