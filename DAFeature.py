

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import Input,Dense, Embedding, LSTM, Bidirectional, GlobalAveragePooling1D, TimeDistributed
from keras.initializers import Constant


from keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import keras.backend as K
from gensim.models import KeyedVectors 

from AttentionLayers import *

from keras.optimizers import SGD, Adam, Adadelta, Adagrad, RMSprop, Adamax, Nadam

from keras.callbacks import ReduceLROnPlateau

from keras.regularizers import l2


abspath = '/home/pnguyen/projects/DAProject'

EMBEDDING_DIM = 300 # how big is each word vector
VOCAB_SIZE = 40000 # how many unique words to use (i.e num rows in embedding vector)
MAX_LENGTH = 100 # max number of words in a comment to use
VALIDATION_SPLIT = 0.3
num_classes = 42

with open(abspath+ '/data/swdaTRAIN.pkl', 'rb') as f:
    df_train = pickle.load(f)
with open(abspath+ '/data/swdaVALID.pkl', 'rb') as f:
    df_valid = pickle.load(f)
with open(abspath+ '/data/swdaTEST.pkl', 'rb') as f:
    df_test = pickle.load(f)

########################################################
def encode(df_train, df_valid, df_test):
    tokenizer_obj = Tokenizer(num_words = VOCAB_SIZE, filters='')  ## ,filters = '([+/\}\[\]]|\{\w),'
    tokenizer_obj.fit_on_texts(list(df_train['text']) + list(df_valid['text']) + list(df_test['text']))
    word_index = tokenizer_obj.word_index
    sequences_train = tokenizer_obj.texts_to_sequences(df_train['text'])
    X_train= pad_sequences(sequences_train, maxlen=MAX_LENGTH )
    sequences_valid = tokenizer_obj.texts_to_sequences(df_valid['text'])
    X_valid = pad_sequences(sequences_valid, maxlen=MAX_LENGTH )
    sequences_test = tokenizer_obj.texts_to_sequences(df_test['text'])
    X_test = pad_sequences(sequences_test, maxlen=MAX_LENGTH )
    return X_train, X_valid, X_test, word_index

######## EMBEDDING MATRIX #######################################
def create_embedding_matrix(glove_path, word_index):   #'/home/phuong/Desktop/glove.6B/glove.6B.300d.txt'
    #word_index = tokenizer_obj.word_index
    nb_words = min(VOCAB_SIZE, len(word_index))+1
    embedding_index = KeyedVectors.load_word2vec_format(glove_path, binary=True) 
  
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= VOCAB_SIZE: 
            continue
        if word in embedding_index:
            #words not found in embedding index will be all-zeros
            embedding_vector = embedding_index.get_vector(word)
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


############ define model ##############################
def Model_RNN_LSTM(embedding_matrix, num_classes):# trainable = True
    model = Sequential()
    embedding_layer = Embedding(VOCAB_SIZE,
                            EMBEDDING_DIM,
                            embeddings_initializer= Constant(embedding_matrix),
                            input_length=MAX_LENGTH,
                            trainable=False)
    model.add(embedding_layer)
    #model.add(LSTM(units=128, dropout=0.3,kernel_initializer='random_uniform', recurrent_initializer='glorot_uniform', return_sequences=True))       
    model.add(Bidirectional(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True,kernel_initializer='random_uniform', recurrent_initializer='glorot_uniform')))  
    #model.add(Bidirectional(LSTM(units=128, dropout=0.5, recurrent_dropout=0.5)))    
    model.add(TimeDistributed(Dense(128, input_shape=(MAX_LENGTH, 128))))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(num_classes, activation='softmax'))
    # try using different optimizers and different optimize cofigs
    #adam = Adam(lr= 1e-3)
    adam = Adam(lr= 1e-3)
    #sgd = SGD(lr = 1e-3, decay = 0.7)
    model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])
    print(model.summary())
    return model

############################################

X_train, X_valid, X_test, word_index = encode(df_train, df_valid, df_test)
 
df_train['label'] = df_train['label'].astype('category')
Y_train = to_categorical(df_train['label'], num_classes = num_classes)

df_valid['label'] = df_valid['label'].astype('category')
Y_valid = to_categorical(df_valid['label'], num_classes = num_classes)

df_test['label'] = df_test['label'].astype('category')
Y_test = to_categorical(df_test['label'], num_classes = num_classes)

embedding_matrix = create_embedding_matrix(abspath + '/wordembedding/GoogleNews-vectors-negative300.bin', word_index)

model = Model_RNN_LSTM(embedding_matrix, num_classes)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001)
history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid)
                          , epochs=30, batch_size= 32, verbose=1, callbacks=[reduce_lr])


abspath2 = '/home/pnguyen/projects/DAProject/FEATUREmainDA'



model.save(abspath2+ '/model.h5')  # creates a HDF5 file 'my_model.h5'


#model = load_model(abspath2+ '/model_KhanpourVATE.h5')
#print(model.summary())
score = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

LAYER_DESIRED = 2
get_layer_output = K.function([model.layers[0].input, K.learning_phase()],[model.layers[LAYER_DESIRED].output])


def genData(df, X):
    dfdlnum = df['conversation_no'].tolist()
    index = df.index
    X = pd.DataFrame(X, index = index)
    dfnum = []
    for i in dfdlnum:
        if i not in dfnum:
            dfnum.append(i)
    
    Xbatch =  []
    Ybatch = []
    for i in dfnum:
        Xi = X.loc[df['conversation_no']== i].values         #.astype(np.float64)
        #Xi.astype(np.float64)
        dfTemp = df.loc[df['conversation_no']== i]
        Yi = dfTemp['label'].values.astype(np.int64) + 1
        Xbatchi = get_layer_output([Xi, 0])[0]
        Xbatch.append(Xbatchi)
        Ybatch.append(Yi)
    Xmul = np.concatenate([fi for fi in Xbatch], axis = 0)
    Ymul = df['label'].values 
    return Xmul, Ymul, Xbatch, Ybatch

Xmul_train, Ymul_train, Xbatch_train, Ybatch_train = genData(df_train, X_train)
Xmul_valid, Ymul_valid, Xbatch_valid, Ybatch_valid = genData(df_valid, X_valid)
Xmul_test, Ymul_test, Xbatch_test, Ybatch_test = genData(df_test, X_test)

########
np.save(abspath2 + '/X_train', X_train)
np.save(abspath2 + '/X_test', X_test)
np.save(abspath2 + '/Y_train', Y_train)
np.save(abspath2 + '/Y_test', Y_test)
np.save(abspath2 + '/Xmul_train', Xmul_train)
np.save(abspath2 + '/Xmul_test', Xmul_test)
np.save(abspath2 + '/Ymul_train', Ymul_train)
np.save(abspath2 + '/Ymul_test', Ymul_test)
np.save(abspath2 + '/Xbatch_train', Xbatch_train)
np.save(abspath2 + '/Xbatch_test', Xbatch_test)
np.save(abspath2 + '/Ybatch_train', Ybatch_train)
np.save(abspath2 + '/Ybatch_test', Ybatch_test)
np.save(abspath2 + '/Ybatch_valid', Ybatch_valid)
np.save(abspath2 + '/Xbatch_valid', Xbatch_valid)
np.save(abspath2 + '/Ymul_valid', Ymul_valid)
np.save(abspath2 + '/Xmul_valid', Xmul_valid)
