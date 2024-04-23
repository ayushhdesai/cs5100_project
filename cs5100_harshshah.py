import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import collections
import os
import re
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("eng_-french.csv")
df.head()

df["French words/sentences"]=("<start> "+df["French words/sentences"]+" <end>")
df["English words/sentences"]=("<start> "+df["English words/sentences"]+" <end>")
df=df.sample(frac=1).reset_index(drop=True)
df

df["French word numbers"]=(df['English words/sentences'].str.split().apply(len))
df["English word numbers"]=(df['French words/sentences'].str.split().apply(len))

data = df[["French word numbers", "English word numbers"]]
sns.boxplot(data=data)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=15, alpha=1)
plt.xlim(0, 40)

def tokenization(x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer

def pad(x, length=14):
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen = length, padding = 'post')

def clean_text(text):
    cleaned_texts=[]
    for sent in text:
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', sent)
        cleaned_texts.append(cleaned_text)
    return cleaned_texts

def preprocess(x, y):
    #cleaned_x=remove_stop(x,"english")
    #cleaned_y=remove_stop(y,"french")

    preprocess_x, x_tk = tokenization(x)
    preprocess_y, y_tk = tokenization(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    return preprocess_x, preprocess_y, x_tk, y_tk

eng = df['English words/sentences']
fr = df['French words/sentences']
preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = preprocess(eng, fr)

max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)

class positional_encoding(tf.keras.layers.Layer):
    def __init__(self,max_sentence_len,embedding_size,**kwargs):
        super().__init__(**kwargs)

        self.pos=np.arange(max_sentence_len).reshape(1,-1).T
        self.i=np.arange(embedding_size/2).reshape(1,-1)
        self.pos_emb=np.empty((1,max_sentence_len,embedding_size))
        self.pos_emb[:,:,0 : :2]=np.sin(self.pos / np.power(10000, (2 * self.i / embedding_size)))
        self.pos_emb[:,:,1 : :2]=np.cos(self.pos / np.power(10000, (2 * self.i / embedding_size)))
        self.positional_embedding = tf.cast(self.pos_emb,dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.positional_embedding

class paddding(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def call(self,inputs):
        mask=1-tf.cast(tf.math.equal(inputs,0),tf.float32)
        return mask[:, tf.newaxis, :]

class look_ahead_mask(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def call(self,sequence_length):
        mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
        return mask

class input_layer_encoder(tf.keras.layers.Layer):
    def __init__(self,max_sentence_len,embedding_size,vocab_size,**kwargs):
        super().__init__(**kwargs)
        self.paddding_mask=paddding()

        self.embedding=tf.keras.layers.Embedding(vocab_size,
                                                 embedding_size,
                                                 input_length=max_sentence_len,
                                                 input_shape=(max_sentence_len,))

        self.positional_encoding=positional_encoding(max_sentence_len,embedding_size)
    def call(self,inputs):
        mask=self.paddding_mask(inputs)

        emb=self.embedding(inputs)

        emb=self.positional_encoding(emb)
        return emb,mask

class input_layer_decoder(tf.keras.layers.Layer):
    def __init__(self,max_sentence_len,embedding_size,vocab_size,**kwargs):
        super().__init__(**kwargs)
        self.paddding_mask=paddding()

        self.embedding=tf.keras.layers.Embedding(vocab_size,
                                                 embedding_size,
                                                 input_length=max_sentence_len,
                                                 input_shape=(max_sentence_len,))

        self.positional_encoding=positional_encoding(max_sentence_len,embedding_size)

        self.look_ahead_mask=look_ahead_mask()
        self.max_sentence_len=max_sentence_len
    def call(self,inputs):
        mask=self.paddding_mask(inputs)

        emb=self.embedding(inputs)

        emb=self.positional_encoding(emb)

        look_head_mak=self.look_ahead_mask(self.max_sentence_len)
        look_head_mak=tf.bitwise.bitwise_and(tf.cast(look_head_mak,dtype=np.int8),tf.cast(mask,dtype=np.int8))
        return emb,look_head_mak

class Encoder_layer(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_size,
                 heads_num,
                 dense_num,
                 dropout_rate=0.0,
                 **kwargs):

        super().__init__(**kwargs)


        self.multi_attention=tf.keras.layers.MultiHeadAttention(
                num_heads=heads_num,
                key_dim=embedding_size,
                dropout=dropout_rate,
            )

        self.Dropout=tf.keras.layers.Dropout(dropout_rate)

        self.ff=tf.keras.Sequential([
            tf.keras.layers.Dense(dense_num,activation="relu"),
            tf.keras.layers.Dense(dense_num,activation="relu"),
            tf.keras.layers.Dense(dense_num,activation="relu"),
            tf.keras.layers.Dense(embedding_size,activation="relu"),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        self.add=tf.keras.layers.Add()

        self.norm1=tf.keras.layers.LayerNormalization()
        self.norm2=tf.keras.layers.LayerNormalization()
    def call(self,inputs,mask,training):

        mha=self.multi_attention(inputs,inputs,inputs,mask)

        norm=self.norm1(self.add([inputs,mha]))

        fc=self.ff(norm)

        A=self.Dropout(fc,training=training)

        output=self.norm2(self.add([A,norm]))

        return output

class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 max_sentence_len,
                 embedding_size,
                 vocab_size,
                 heads_num,
                 dense_num,
                 num_of_encoders,
                 **kwargs):
        super().__init__(**kwargs)
        self.add=tf.keras.layers.Add()
        self.input_layer=input_layer_encoder(max_sentence_len,embedding_size,vocab_size)
        self.encoder_layer=[Encoder_layer(embedding_size,heads_num, dense_num) for i in range (num_of_encoders)]
        self.num_layers=num_of_encoders
    def call(self,inputs,training):
        emb,mask=self.input_layer(inputs)
        skip=emb
        for layer in self.encoder_layer:
            emb = layer(emb, mask,training)
            emb = self.add([skip,emb])
            skip = emb
        return emb,mask

class decoder_layer(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_size,
                 heads_num,
                 dense_num,
                 dropout_rate=0.0,
                 **kwargs):

        super().__init__(**kwargs)

        self.masked_mha=tf.keras.layers.MultiHeadAttention(
                num_heads=heads_num,
                key_dim=embedding_size,
                dropout=dropout_rate,
            )


        self.multi_attention=tf.keras.layers.MultiHeadAttention(
                num_heads=heads_num,
                key_dim=embedding_size,
                dropout=dropout_rate,
            )

        self.ff=tf.keras.Sequential([
            tf.keras.layers.Dense(dense_num,activation="relu"),
            tf.keras.layers.Dense(dense_num,activation="relu"),
            tf.keras.layers.Dense(dense_num,activation="relu"),
            tf.keras.layers.Dense(embedding_size,activation="relu"),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        self.Dropout=tf.keras.layers.Dropout(dropout_rate)
        self.add=tf.keras.layers.Add()

        self.norm1=tf.keras.layers.LayerNormalization()
        self.norm2=tf.keras.layers.LayerNormalization()
        self.norm3=tf.keras.layers.LayerNormalization()

    def call(self,inputs,encoder_output,enc_mask,look_head_mask,training):

        mha_out,atten_score=self.masked_mha(inputs,inputs,inputs,look_head_mask,return_attention_scores=True)

        Q1=self.norm1(self.add([inputs,mha_out]))

        mha_out2,atten_score2=self.multi_attention(Q1,encoder_output,encoder_output,enc_mask,return_attention_scores=True)

        Z=self.norm2(self.add([Q1,mha_out2]))

        fc =  self.ff(Z)

        A=self.Dropout(fc,training=training)

        output=self.norm3(self.add([A,Z]))
        return output

class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 max_sentence_len,
                 embedding_size,
                 vocab_size,
                 heads_num,
                 dense_num,
                 num_of_decoders,
                 **kwargs):
        super().__init__(**kwargs)
        self.add=tf.keras.layers.Add()
        self.input_layer=input_layer_decoder(max_sentence_len,embedding_size,vocab_size)
        self.decoder_layer=[decoder_layer(embedding_size,heads_num, dense_num) for i in range (num_of_decoders)]
        self.num_layers=num_of_decoders
    def call(self,inputs,encoder_output,enc_mask,training):
        emb,look_head_mask=self.input_layer(inputs)
        skip=emb
        for layer in self.decoder_layer:
            emb = layer(emb,encoder_output,enc_mask,look_head_mask,training)
            emb = self.add([skip,emb])
            skip = emb
        return emb

class transformer(tf.keras.Model):
    def __init__(self,
                 max_sentence_len_1=None,
                 max_sentence_len_2=None,
                 embedding_size=None,
                 vocab_size1=None,
                 vocab_size2=None,
                 heads_num=None,
                 dense_num=None,
                 num_of_encoders_decoders=None):

        super(transformer,self).__init__()

        self.Encoder=Encoder(max_sentence_len_1,
                             embedding_size,
                             vocab_size1,
                             heads_num,
                             dense_num,
                             num_of_encoders_decoders)

        self.Decoder=Decoder(max_sentence_len_2,
                             embedding_size,
                             vocab_size2,
                             heads_num,
                             dense_num,
                             num_of_encoders_decoders,)

        self.Final_layer=tf.keras.layers.Dense(vocab_size2, activation='relu')

        self.softmax=tf.keras.layers.Softmax(axis=-1)
    def call(self, inputs):
        input_sentence,output_sentence=inputs
        enc_output,enc_mask=self.Encoder(input_sentence)

        dec_output=self.Decoder(output_sentence,enc_output,enc_mask)

        final_out=self.Final_layer(dec_output)

        softmax_out=self.softmax(final_out)
        return softmax_out

tran=transformer(max_sentence_len_1=14,
                     max_sentence_len_2=13,
                     embedding_size=300,
                     vocab_size1=french_vocab_size+1,
                     vocab_size2=english_vocab_size+1,
                     heads_num=5,
                     dense_num=512,
                     num_of_encoders_decoders=1)

tran((preproc_french_sentences[:1],preproc_english_sentences[:1,:-1]))
tran.summary()

tran.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
             optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
             metrics=["accuracy"])

history=tran.fit((preproc_french_sentences,preproc_english_sentences[:,:-1]),
         preproc_english_sentences[:,1:,tf.newaxis],
         epochs=30, verbose = True,
         batch_size=512)

def prepare_pred(sent):
    output=english_tokenizer.texts_to_sequences(sent)
    output=pad(output,13)
    return output

def pred(i):
    sent=["<start>"]
    french_token=prepare_pred(sent)
    word=np.argmax(tran.predict((preproc_french_sentences[[i]],french_token),verbose=0),-1)[0,0]
    sent[0]=sent[0]+ " "+english_tokenizer.sequences_to_texts(np.array([[word]]))[0]
    for j in range(1,12):
        french_token=prepare_pred(sent)
        word=np.argmax(tran.predict((preproc_french_sentences[[i]],french_token),verbose=0),-1)[0,j]
        sent[0]=sent[0]+ " "+english_tokenizer.sequences_to_texts(np.array([[word]]))[0]
        if english_tokenizer.sequences_to_texts(np.array([[word]]))[0]=="end":
            break
    return sent

import random
def show():
    i=random.randint(0,170111)
    print("french sent : ",french_tokenizer.sequences_to_texts(preproc_french_sentences[[i]]))
    print("predict sent : ",pred(i))
    print("true sent : ",english_tokenizer.sequences_to_texts(preproc_english_sentences[[i]]))

for i in range(5):
    show()
    print("----------------")

import numpy as np
import random
from nltk.translate.bleu_score import sentence_bleu

# Define the prepare_pred function
def prepare_pred(sent):
    output = english_tokenizer.texts_to_sequences(sent)
    output = pad(output, 13)
    return output

# Define the pred function
def pred(i):
    sent = ["<start>"]
    french_token = prepare_pred(sent)
    word = np.argmax(tran.predict((preproc_french_sentences[[i]], french_token), verbose=0), -1)[0, 0]
    sent[0] = sent[0] + " " + english_tokenizer.sequences_to_texts(np.array([[word]]))[0]
    for j in range(1, 12):
        french_token = prepare_pred(sent)
        word = np.argmax(tran.predict((preproc_french_sentences[[i]], french_token), verbose=0), -1)[0, j]
        sent[0] = sent[0] + " " + english_tokenizer.sequences_to_texts(np.array([[word]]))[0]
        if english_tokenizer.sequences_to_texts(np.array([[word]]))[0] == "end":
            break
    return sent

# Define the show function
def show():
    i = random.randint(0, 170111)
    french_sent = french_tokenizer.sequences_to_texts(preproc_french_sentences[[i]])[0]
    pred_sent = pred(i)[0]
    true_sent = english_tokenizer.sequences_to_texts(preproc_english_sentences[[i]])[0]
    bleu_score = sentence_bleu([true_sent.split()], pred_sent.split())  # Calculate BLEU score
    print("French sent : ", french_sent)
    print("Predicted sent : ", pred_sent)
    print("True sent : ", true_sent)
    print("BLEU Score:", bleu_score)

# Call the show function to display predictions and calculate BLEU scores
for i in range(5):
    show()
    print("----------------")


# Define the prepare_pred function
def prepare_pred(sent):
    output = english_tokenizer.texts_to_sequences(sent)
    output = pad(output, 13)
    return output

# Define the pred function
def pred(i):
    sent = ["<start>"]
    french_token = prepare_pred(sent)
    word = np.argmax(tran.predict((preproc_french_sentences[[i]], french_token), verbose=0), -1)[0, 0]
    sent[0] = sent[0] + " " + english_tokenizer.sequences_to_texts(np.array([[word]]))[0]
    for j in range(1, 12):
        french_token = prepare_pred(sent)
        word = np.argmax(tran.predict((preproc_french_sentences[[i]], french_token), verbose=0), -1)[0, j]
        sent[0] = sent[0] + " " + english_tokenizer.sequences_to_texts(np.array([[word]]))[0]
        if english_tokenizer.sequences_to_texts(np.array([[word]]))[0] == "end":
            break
    return sent

# Define the show function
def show():
    i = random.randint(0, 170111)
    french_sent = french_tokenizer.sequences_to_texts(preproc_french_sentences[[i]])[0]
    pred_sent = pred(i)[0]
    true_sent = english_tokenizer.sequences_to_texts(preproc_english_sentences[[i]])[0]
    bleu_score = sentence_bleu([true_sent.split()], pred_sent.split())  # Calculate BLEU score
    return bleu_score

# Calculate average BLEU score
total_bleu_score = 0.0
num_samples = 1000  # Adjust this based on the size of your dataset
for _ in range(num_samples):
    total_bleu_score += show()

average_bleu_score = total_bleu_score / num_samples
print("Average BLEU Score for", num_samples, "samples:", average_bleu_score)

tran1=transformer(max_sentence_len_1=14,
                     max_sentence_len_2=13,
                     embedding_size=300,
                     vocab_size1=french_vocab_size+1,
                     vocab_size2=english_vocab_size+1,
                     heads_num=5,
                     dense_num=512,
                     num_of_encoders_decoders=1)

tran1((preproc_french_sentences[:1],preproc_english_sentences[:1,:-1]))
tran1.summary()

tran1.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
             optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
             metrics=["accuracy"])

history1=tran1.fit((preproc_french_sentences,preproc_english_sentences[:,:-1]),
         preproc_english_sentences[:,1:,tf.newaxis],
         epochs=30, verbose = True,
         batch_size=512)


# Define the prepare_pred function
def prepare_pred(sent):
    output = english_tokenizer.texts_to_sequences(sent)
    output = pad(output, 13)
    return output

# Define the pred function
def pred(i):
    sent = ["<start>"]
    french_token = prepare_pred(sent)
    word = np.argmax(tran1.predict((preproc_french_sentences[[i]], french_token), verbose=0), -1)[0, 0]
    sent[0] = sent[0] + " " + english_tokenizer.sequences_to_texts(np.array([[word]]))[0]
    for j in range(1, 12):
        french_token = prepare_pred(sent)
        word = np.argmax(tran1.predict((preproc_french_sentences[[i]], french_token), verbose=0), -1)[0, j]
        sent[0] = sent[0] + " " + english_tokenizer.sequences_to_texts(np.array([[word]]))[0]
        if english_tokenizer.sequences_to_texts(np.array([[word]]))[0] == "end":
            break
    return sent

# Define the show function
def show():
    i = random.randint(0, 170111)
    french_sent = french_tokenizer.sequences_to_texts(preproc_french_sentences[[i]])[0]
    pred_sent = pred(i)[0]
    true_sent = english_tokenizer.sequences_to_texts(preproc_english_sentences[[i]])[0]
    bleu_score = sentence_bleu([true_sent.split()], pred_sent.split())  # Calculate BLEU score
    return bleu_score

# Calculate average BLEU score
total_bleu_score = 0.0
num_samples = 1000  # Adjust this based on the size of your dataset
for _ in range(num_samples):
    total_bleu_score += show()

average_bleu_score = total_bleu_score / num_samples
print("Average BLEU Score for", num_samples, "samples:", average_bleu_score)

tran2=transformer(max_sentence_len_1=14,
                     max_sentence_len_2=13,
                     embedding_size=300,
                     vocab_size1=french_vocab_size+1,
                     vocab_size2=english_vocab_size+1,
                     heads_num=5,
                     dense_num=512,
                     num_of_encoders_decoders=1)

tran2((preproc_french_sentences[:1],preproc_english_sentences[:1,:-1]))
tran2.summary()

tran2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
             optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-2),
             metrics=["accuracy"])

history2=tran2.fit((preproc_french_sentences,preproc_english_sentences[:,:-1]),
         preproc_english_sentences[:,1:,tf.newaxis],
         epochs=30, verbose = True,
         batch_size=512)


# Define the prepare_pred function
def prepare_pred(sent):
    output = english_tokenizer.texts_to_sequences(sent)
    output = pad(output, 13)
    return output

# Define the pred function
def pred(i):
    sent = ["<start>"]
    french_token = prepare_pred(sent)
    word = np.argmax(tran2.predict((preproc_french_sentences[[i]], french_token), verbose=0), -1)[0, 0]
    sent[0] = sent[0] + " " + english_tokenizer.sequences_to_texts(np.array([[word]]))[0]
    for j in range(1, 12):
        french_token = prepare_pred(sent)
        word = np.argmax(tran2.predict((preproc_french_sentences[[i]], french_token), verbose=0), -1)[0, j]
        sent[0] = sent[0] + " " + english_tokenizer.sequences_to_texts(np.array([[word]]))[0]
        if english_tokenizer.sequences_to_texts(np.array([[word]]))[0] == "end":
            break
    return sent

# Define the show function
def show():
    i = random.randint(0, 170111)
    french_sent = french_tokenizer.sequences_to_texts(preproc_french_sentences[[i]])[0]
    pred_sent = pred(i)[0]
    true_sent = english_tokenizer.sequences_to_texts(preproc_english_sentences[[i]])[0]
    bleu_score = sentence_bleu([true_sent.split()], pred_sent.split())  # Calculate BLEU score
    return bleu_score

# Calculate average BLEU score
total_bleu_score = 0.0
num_samples = 1000  # Adjust this based on the size of your dataset
for _ in range(num_samples):
    total_bleu_score += show()

average_bleu_score = total_bleu_score / num_samples
print("Average BLEU Score for", num_samples, "samples:", average_bleu_score)

#RMS Prop vs Adam (Accuracy)
# Plot accuracy for Adam Optimizer
plt.plot(history1.history['accuracy'], 'b', label='Root Mean Square Propagation (LR=0.001)')

# Plot accuracy for RMSprop
plt.plot(history.history['accuracy'], 'r', label='Adam (LR=0.001)')

# Set title and labels
plt.title("Accuracy Comparison for Different Optimizers")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Add legend
plt.legend()

# Show plot
plt.show()

#RMS Prop vs Adam (Loss)
# Plot loss for Adam Optimizer
plt.plot(history1.history['loss'], 'b', label='Root Mean Square Propagation (LR=0.001)')

# Plot loss for RMSprop
plt.plot(history.history['loss'], 'r', label='Adam (LR=0.001)')

# Set title and labels
plt.title("Loss Comparison for Different Optimizers")
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Add legend
plt.legend()

# Show plot
plt.show()

#RMS Prop vs RMS Prop (Accuracy)
# Plot accuracy for Adam Optimizer
plt.plot(history1.history['accuracy'], 'b', label='Root Mean Square Propagation (LR=0.001)')

# Plot accuracy for RMSprop
plt.plot(history2.history['accuracy'], 'r', label='Root Mean Square Propagation (LR=0.01)')

# Set title and labels
plt.title("Accuracy Comparison for Different Optimizers")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Add legend
plt.legend()

# Show plot
plt.show()

#RMS Prop vs RMS Prop (loss)
# Plot loss for Adam Optimizer
plt.plot(history1.history['loss'], 'b', label='Root Mean Square Propagation (LR=0.001)')

# Plot loss for RMSprop
plt.plot(history2.history['loss'], 'r', label='Root Mean Square Propagation (LR=0.01)')

# Set title and labels
plt.title("Loss Comparison for Different Optimizers")
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Add legend
plt.legend()

# Show plot
plt.show()

