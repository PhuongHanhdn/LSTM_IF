""" Created on Wed Jan  2 00:32:55 2019

@author: Phuong Hanh Tran
"""
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import tensorflow as tf
from keras import optimizers, Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
from numpy import arange, sin, pi, random
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import scipy.integrate as integrate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest
from scipy.stats import pearsonr

from numpy.random import seed
tf.random.set_seed(11)

SEED = 123 #used to help randomly select the data points
DATA_SPLIT_PCT = 0.2
# Global hyper-parameters
sequence_length = 11
epochs = 30
batch_size = 50
lr = 0.01

def gen_wave():
    """ Generate a synthetic wave by adding up a few sine waves and some noise
    :return: the final wave
    """
    t = np.arange(0.0, 20.0, 0.01)
    wave1 = sin(2 * 2 * pi * t)
    noise = random.normal(0, 0.2, len(t))
    wave1 = wave1 + noise
    print("wave1", len(wave1))
    wave2 = sin(2 * pi * t)
    print("wave2", len(wave2))
    t_rider = arange(0.0, 0.5, 0.01)
    wave3 = -2*sin(10 * pi * t_rider)
    print("wave3", len(wave3))
    insert = round(0.8 * len(t))
    wave1[insert:insert + 50] = wave1[insert:insert + 50] + wave3
    #wave1[100:150] = wave1[100:150] + wave3
    return wave1 - 2*wave2


def z_norm(result):
    result_mean = result.mean()
    result_std = result.std()
    result -= result_mean
    result /= result_std
    return result, result_mean


def get_split_prep_data(train_start, train_end,
                          test_start, test_end):
    data = gen_wave()
    print("Length of Data", len(data))

    # train data
    print ("Creating train data...")

    result = []
    for index in range(train_start, train_end - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    result, result_mean = z_norm(result)

    print ("Mean of train data : ", result_mean)
    print ("Train data shape  : ", result.shape)

    train = result[train_start:train_end, :]
    np.random.shuffle(train)  # shuffles in-place
    X_train = train[:, :-1]
    y_train = train[:, -1]

    # test data
    print ("Creating test data...")

    result = []
    for index in range(test_start, test_end - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    result, result_mean = z_norm(result)

    print ("Mean of test data : ", result_mean)
    print ("Test data shape  : ", result.shape)

    X_test = result[:, :-1]
    y_test = result[:, -1]

    print("Shape X_train", np.shape(X_train))
    print("Shape X_test", np.shape(X_test))

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test


# train on first 1400 samples and test on next 600 samples (has anomaly)

X_train,y_train, X_test, y_test = get_split_prep_data(0, 1399, 1400, 2000)
X_train_X, X_valid = train_test_split(X_train, test_size=DATA_SPLIT_PCT, random_state=SEED)
timesteps =  X_train.shape[1] # equal to the sequence_length
n_features =  X_train.shape[2] # 1

lstm_autoencoder = Sequential()

# Encoder
lstm_autoencoder.add(LSTM(512, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
lstm_autoencoder.add(LSTM(64, activation='relu', return_sequences=False))
lstm_autoencoder.add(RepeatVector(timesteps))
# Decoder
lstm_autoencoder.add(LSTM(64, activation='relu', return_sequences=True))
lstm_autoencoder.add(LSTM(512, activation='relu', return_sequences=True))
lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

lstm_autoencoder.summary()

adam = optimizers.Adam(lr)
lstm_autoencoder.compile(loss='mse', optimizer=adam)

cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",
                               save_best_only=True,
                               verbose=0)

tb = TensorBoard(log_dir='./logs',
                histogram_freq=0,
                write_graph=True,
                write_images=True)

lstm_autoencoder_history = lstm_autoencoder.fit(X_train, X_train, 
                                                epochs=epochs, 
                                                batch_size=batch_size, 
                                                validation_data=(X_valid, X_valid),
                                                verbose=2).history
                                                
plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train')
plt.plot(lstm_autoencoder_history['val_loss'], linewidth=2, label='Valid')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

def flatten(X):
    '''
    Flatten a 3D array.
    
    Input
    X            A 3D array for lstm, where the array is sample x sequence length x features.
    
    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)

print("Predicting...")
predicted_train = lstm_autoencoder.predict(X_train)
predicted = lstm_autoencoder.predict(X_test)
#print("Reshaping predicted")
#predicted_train = np.reshape(predicted_train, (predicted_train.size,))
#predicted = np.reshape(predicted, (predicted.size,))

plt.figure(1)
plt.title("Actual Test Signal w/Anomalies")
plt.plot(y_train[:len(y_train)], 'b')

plt.figure(2)
plt.title("Actual Test Signal w/Anomalies")
plt.plot(y_test[:len(y_test)], 'b')

plt.figure(3)
plt.title("Squared Error")
mse = np.mean(np.power(flatten(X_test) - flatten(predicted), 2), axis=1)
plt.plot(mse, 'r')
mse_train = np.mean(np.power(flatten(X_train) - flatten(predicted_train), 2), axis=1)

y_test1=np.ones(X_test.shape[0])
y_test1[199:249]=-1


   
#IF

# fit the model
e=X_train - predicted_train
nsamples, nx, ny = e.shape
d2_e = e.reshape((nsamples,nx*ny))
rng = np.random.RandomState(1000)
clf = IsolationForest(max_samples=10000, random_state=rng)
clf.fit(d2_e)
e_t=X_test - predicted
nsamples, nx, ny = e_t.shape
d2_e_t = e_t.reshape((nsamples,nx*ny))
y_scores = clf.predict(d2_e_t)
precision = precision_score(y_test1, y_scores)
recall    = recall_score(y_test1, y_scores)
accuracy = accuracy_score(y_test1, y_scores)
f1 = f1_score(y_test1, y_scores, average='macro')
print ('Precision : ', precision)
print ('Recall: ', recall)
print ('Accuracy : ', accuracy)
print ('F1_score: ', f1)


import pandas as pd
from sklearn.decomposition import PCA
pca = PCA(2)
nsamples, nx, ny = e_t.shape
e_tx = e_t.reshape((nsamples,nx*ny))
pca.fit(e_tx)
res=pd.DataFrame(pca.transform(e_tx))
Z = np.array(res)

x_min, x_max = res[0].min() - .5, res[0].max() + .5
y_min, y_max = res[1].min() - .5, res[1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(res[0], res[1], c=y_scores, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Principle component 1')
plt.ylabel('Principle component 2')
plt.title("Anomaly detection using combination of LSTM Autoencoder and Isolation Forest")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())


