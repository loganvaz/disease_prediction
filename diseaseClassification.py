###dataset taken from https://www.kaggle.com/kaushil268/disease-prediction-using-machine-learning?select=Training.csv
#I have both of these excel sheets in the same folder as this one is

#for file creating and saving
import os
import tensorflow as tf

#inputs
loadModel = False#should only be used if don't have a val split or use random seed for it (thought coding that would be eventually necessary, but doesn't look like that's the case)
lr = 0.01

#data_processing/model
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical

#for model
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Activation

def predictionModel(inputSize, outputSize):#because data is all one_hot, no need to normalize
    X = Input(shape=(inputSize))

    Theta = Dense(216, activation = "relu")(X)

    Theta = Dense(64, activation = "relu")(X)

    Theta = Dropout(0.15)(Theta)

    Theta2 = Dense(128, activation = "relu")(X)

    Theta = Activation("relu")(Dense(64)(Theta2) + Theta)

    Theta = Dropout(0.15)(Theta)

    predictions = Dense(outputSize, activation="softmax")(Theta)

    model =  Model(inputs = X, outputs = predictions)

    return model

#creates folders
def createFolders(h, n):
    try:
        f = os.mkdir(h+"\\"+n)
        f.close()
    except:
        return h+"\\"+n
    return h+"\\"+n

#checkpoint to save (note that this shouldn't be used with random shuffle of validation data, if this was required should have seperate program that divides beforehand or send random seed)
filepath = createFolders(os.getcwd(), "modelInProgress")+"\\modelInProgess"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

    

#class to convert from text data to one_hot and back again
class y_convert:
    listed = list()
    invDict = dict()
    def __init__(self, y_text):
        self.listed = list(set(y_text))
        numClasses = len(self.listed)
        for i in range(numClasses):
            self.invDict[self.listed[i]] = i

        
    def get_one_hot(self, y):
        y = np.array([self.invDict[i] for i in y])
        #toReturn = np.zeros((len(y), len(self.listed)))
        #toReturn[np.arange(len(y), y)] = 1
        return to_categorical(y)
        #toReturn = np.zeros((len(y), len(listed)))
        #for i in range(len(y)):
    def getText(self, y_hat):
        try:#it's prediction
            return [self.listed[i] for i in y_hat]
        except:#it's one_hot
            return [self.listed[np.argmax(i)] for i in y_hat]

#f1, additional metric to judge accuracy (if no disease was option, this probably would have been necessarily due to data imbalance)
import keras.backend as K        
def f1(y_true, y_pred):#taken from old keras source code except for the epsilon removal
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives )
    recall = true_positives / (possible_positives )
    f1_val = 2*(precision*recall)/(precision+recall)
    return f1_val
    
            
                
        
        

#loading training data
dataset = pd.read_csv("Training.csv")

dataset = dataset.sample(dataset.shape[0])#shuffling the dataSet so can divide into train and val set

dataset = np.array(dataset)

dataset = dataset[:, :-1]#getting rid of errant in training
#dividing X and y data
y = dataset[:, -1]
X = dataset[:, :-1]


converter = y_convert(y)#class to convert

y = converter.get_one_hot(y)


model = predictionModel(X.shape[1], y.shape[1])

if (loadModel):
    try:
        
        model.load_weights(filepath)
        print("succesfully loaded model")
    except:
        print("have failed to load weights for model")
print(model.summary())

optimizer = Adam(learning_rate = lr)

model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics = ["accuracy", f1])
cutOff = int(X.shape[0] /2)#val Size is quite large (would normally make smaller, but training is going very well already)

#converting to float so can process
X = np.array(X, dtype="float32")
y = np.array(y, dtype="float32")

#splitting data
X_train, X_val = X[cutOff:], X[:cutOff]
y_train, y_val = y[cutOff:], y[:cutOff]

print("preTraining metrics on train and val (NaN on val is disabled as I took that regularizer away since I'm not optimizing base don it)")
print(model.evaluate(X_train, y_train, verbose=0))
print(model.evaluate(X_val, y_val, verbose=0))

#below was to make sure no overlap
#print(str(X.shape[0]) + " should be the same as " + str(X_train.shape[0] + X_val.shape[0]))




model.fit(X_train,y_train, verbose=2, callbacks = [checkpoint], validation_data = (X_val, y_val), epochs=4)

#testing calculations

dataset_test = np.array(pd.read_csv("Testing.csv"))



X_test = np.array(dataset_test[:, :-1], dtype="float32")
y_test = np.array(converter.get_one_hot(dataset_test[:, -1]), dtype="float32")

print(model.evaluate(X_train, y_train, verbose=0))
