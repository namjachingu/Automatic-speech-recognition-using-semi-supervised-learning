from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.callbacks import Callback
from keras.utils import to_categorical 
from keras.models import Model, load_model
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as mtick

import itertools

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pickle
import pandas as pd
import random
import sys
from matplotlib import colors

#Show full numpy array
np.set_printoptions(threshold=sys.maxsize)

#We have the saved pickle file, now we need to access the pickled file:
# open a file, where you stored the pickled data
file = open('mfcc_train.pckl', 'rb')

# dump information to that file
mfcc_train = pickle.load(file)
# close the file
file.close()

file = open('mfcc_test.pckl', 'rb')
mfcc_test = pickle.load(file)
file.close()

file = open('mfcc_dev.pckl', 'rb')
mfcc_val = pickle.load(file)
file.close()


#Monophones
#Targets for the training data:
targets = {}
nTargets = 0
for n in range(1, 31):
    with open('mono_ali.'+str(n)+'.pdf.txt') as f:
        monoali1 = [x.strip() for x in f.readlines()]
    for item in monoali1:
        #print(item)
        data = item.split() #Split a string into a list where each word is a list item
        #print(data)
        numdata = np.array([int(el) for el in data[1:]]) #here each words become a item
        #print(numdata)
        targets[data[0]] = numdata
        nTargets = np.max([nTargets, numdata.max()])

nTargets += 1
#print(len(mfcc_train['mvjh0_si1556']))

devTargets = {}
dev_nTargets = 0

for n in range(1,2):
    #Targets for the validation data: 
    with open('mono_ali_dev.'+str(n)+'.pdf.txt') as f:
        monoali_dev = [x.strip() for x in f.readlines()]
    for item in monoali_dev:
        data_dev = item.split() #Split a string into a list where each word is a list item
        numdata_dev = np.array([int(el) for el in data_dev[1:]]) #here each words become a item
        devTargets[data_dev[0]] = numdata_dev
        dev_nTargets = np.max([dev_nTargets, numdata_dev.max()])

dev_nTargets  += 1

#Targets for the test data: 
testTargets = {}
test_nTargets = 0
for n in range(1,2):
    with open('mono_ali_test.'+str(n)+'.pdf.txt') as f:
        monoali_test = [x.strip() for x in f.readlines()]
    for item in monoali_test:
        data_test = item.split() #Split a string into a list where each word is a list item
        numdata_test = np.array([int(el) for el in data_test[1:]]) #here each words become a item
        testTargets[data_test[0]] = numdata_test
        test_nTargets = np.max([test_nTargets, numdata_test.max()])
test_nTargets += 1

for keys in mfcc_test.keys():
    targetsarray_test = testTargets[keys]

'''
#Checking if the dictionaries have the same keys
print(testTargets.keys() == targets.keys()) #False
print(devTargets.keys() == targets.keys()) #False
print(devTargets.keys() == testTargets.keys()) #False
'''


phonemes = ['sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'cl', 'd', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'vcl', 'w', 'y', 'z', 'zh']

#print(len(targets['mzmb0_sx86']))
#print(len(tritargets['mzmb0_sx86']))
#print(len(mfcc_train['mzmb0_sx86']))

#print(len(targets['mzmb0_sx446']))
#print(len(tritargets['mzmb0_sx446']))
#print(len(mfcc_train['mzmb0_sx446']))


states = {'sil': [0, 1, 2], 'aa': [3, 4, 5], 'ae': [6, 7, 8], 'ah': [9, 10, 11], 'ao': [12, 13, 14], 'aw': [15, 16, 17],  'ax': [18, 19, 20], 'ay': [21, 22, 23], 'b': [24, 25, 26], 'ch': [27, 28, 29], 'cl': [30, 31, 32], 'd': [33, 34, 35], 'dh': [36, 37, 38], 'dx': [39, 40, 41], 'eh': [42, 43, 44], 'el': [45, 46, 47], 'en': [48, 49, 50], 'epi': [51, 52, 53], 'er': [54, 55, 56], 'ey': [57, 58, 59], 'f': [60, 61, 62], 'g': [63, 64, 65], 'hh': [66, 67, 68], 'ih': [69, 70, 71], 'ix': [72, 73, 74], 'iy': [75, 76, 77], 'jh': [78, 79, 80], 'k': [81, 82, 83], 'l': [84, 85, 86], 'm': [87, 88, 89], 'n': [90, 91, 92], 'ng': [93, 94, 95], 'ow': [96, 97, 98], 'oy': [99, 100, 101], 'p': [102, 103, 104], 'r': [105, 106, 107], 's': [108, 109, 110], 'sh': [111, 112, 113], 't': [114, 115, 116], 'th': [117, 118, 119], 'uh': [120, 121, 122], 'uw': [123, 124, 125], 'v': [126, 127, 128], 'vcl': [129, 130, 131], 'w': [132, 133, 134], 'y': [135, 136, 137], 'z': [138, 139, 140], 'zh': [141, 142, 143]}


#nOut = 1 # ---> loss = 'binary_crossentropy' and activation = 'sigmoid' , 2-class problem
#nOut = number of phones / classes # ---> loss = 'categorical_crossentropy' and activation = 'softmax', Multi-class problem

def frameConcat(x,splice, splType):
    validFrm = int( np.sum(np.sign( np.sum( np.abs(x), axis=1) )) )
    nFrame, nDim = x.shape

    if ( splType == 1):
        spl = splice
        splVec = np.arange(0, int(2*spl+1), 1)
    else:
        spl = int(2*splice)
        splVec = np.arange(0, int(2*spl+1), 2)

    xZerosPad = np.vstack([np.zeros((spl, nDim)), x[0:validFrm,:], np.zeros((spl, nDim))])
    xConc = np.zeros( (validFrm, int(nDim*(2*splice+1))) )

    for iFrm in range(validFrm):
        xConcTmp = np.reshape(xZerosPad[iFrm+splVec,:], (1,int((2*splice+1)*nDim)) )
        xConc[iFrm, :] = xConcTmp
    return xConc

'''
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(143,)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dense(nTargets, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
'''

model = Sequential()
model.add(Dense(143, activation='relu', input_shape=(143,)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dense(nTargets, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


#13 MFCC:
x_train = np.zeros((0, 143)) #13*11, 5 frames on each side of the current mfcc
x_test = np.zeros((0, 143))
x_val = np.zeros((0, 143))
y_train = np.zeros((0, nTargets))
y_test = np.zeros((0, test_nTargets))
y_val = np.zeros((0, dev_nTargets))


utterances = list(mfcc_train.keys())
#random_keys=random.sample(utterances,3696)
#Choosing percentage of training data
#1% of 3696 = 37 utterances
#3% of 3696 = 111 utterances
#5% of  3696 = 185 utterances
#10% of 3696 = 367 utterances
#20% of 3696 = 739 utterances
#30% of 3696 = 1109 utterances

random.seed(0)
#random_keys=random.sample(utterances, 37)
#random_keys=random.sample(utterances, 111)
#random_keys=random.sample(utterances, 185)
#random_keys=random.sample(utterances, 367)
#random_keys=random.sample(utterances, 739)
#random_keys=random.sample(utterances, 1109)
#print(len(random_keys))


#Monophones:
#for keys in random_keys:
for keys in mfcc_train.keys():
    mfccarray = mfcc_train[keys]
    x_mean = np.mean(mfccarray, axis=0)
    x_std = np.std(mfccarray, axis=0)
    mfcctrain_normalized = ( mfccarray - x_mean ) / x_std
    trainConc=frameConcat(mfcctrain_normalized, 5, 1) #should give 13*11
    x_train = np.vstack((x_train, trainConc)) #concatenate mfcc
    
    targetsarray = targets[keys]
    #numberOfClasses = np.max(targetsarray)+1
    Labels = np.eye(nTargets)
    targetOneHot = Labels[targetsarray, :]
    y_train = np.vstack((y_train, targetOneHot)) #concatenated targets


for keys in mfcc_val.keys():
    valarray = mfcc_val[keys]
    mfccval_normalized = ( valarray - x_mean ) / x_std
    valConc=frameConcat(mfccval_normalized, 5, 1) 
    x_val = np.vstack((x_val, valConc)) 

    targetsarray_val = devTargets[keys]
    Labels_val = np.eye(dev_nTargets)
    targetOneHot_val = Labels_val[targetsarray_val, :]
    y_val = np.vstack((y_val, targetOneHot_val)) 


for keys in mfcc_test.keys():
    testarray = mfcc_test[keys]
    mfcctest_normalized = ( testarray - x_mean ) / x_std
    testConc=frameConcat(mfcctest_normalized, 5, 1) 
    x_test = np.vstack((x_test, testConc)) 

    targetsarray_test = testTargets[keys]
    Labels_test = np.eye(test_nTargets)
    targetOneHot_test = Labels_test[targetsarray_test, :]
    y_test = np.vstack((y_test, targetOneHot_test)) 


#Trainig with early stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=256, epochs=20, callbacks=[callback], verbose=1, shuffle=True)
numberOfEpochs = len(history.history['loss'])

'''
fig, ax = plt.subplots()
plt.plot(history.history['accuracy'], marker="*", label='Train accuracy', color='blue')
plt.plot(history.history['val_accuracy'], marker="o", label='Test accuracy', color='red')
plt.title('Frame accuracy rate' )
plt.ylabel('Accuracy rate (%)')
plt.xlabel('Epoch')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=None, symbol=None))
#ax.grid()
plt.legend(loc='best')
plt.show()
plt.savefig('MONOSS20.png')
'''

#Test model: (on full data set)
score, acc = model.evaluate(x_test, y_test, batch_size=256, verbose=1)
print('Test score:', score)
print('Test accuracy:', acc)


##################################### Changing from 3-state error to phoneme error ##################################### 

### Converting one-hot encoded targets back to integer values. Then comparing the int values to class labels of the output from the DNN (y_class)
#targetClass = np.where(y_test==1)[1]
#predictedClass = model.predict_classes(x_test)  #x_test????


targetClass = np.where(y_test==1)[1]
#This works since np.where(array==1)[1] returns the column indices of the 1's, which are exactly the labels.
predictedClass = model.predict_classes(x_test) 

#MAKES LIST OF LIST FROM DICTIONARY
statesValues = []
for keys in states.keys():
    phonemeStates = states[keys] #this is a list
    statesValues.append(phonemeStates)





################## PHONEME RECOGNITION ##################

mapedState = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47] 


######################### I've made two ways of checking whether the predicted states belong to the same phoneme as the target states, both methods are correct. 

###Method 1:
stateInPhoneme = []
counter = 0
for i in range(len(predictedClass)):
    if predictedClass[i] != targetClass[i]:
        #print(mapedState[predictedClass[i]])
        if mapedState[predictedClass[i]] == mapedState[targetClass[i]]:
            #stateInPhoneme.append(i)
            counter += 1

'''
###Method 2:
correctPhonemePosition = []
stateInPhoneme = []
for i in range(len(y_test)):
    if (predictedClass[i] != targetClass[i]):
        #print(any((y_class[i] in x and OHEtoInt[i] in x) for x in newArray))
        for index, nested_list in enumerate(newArray):
            if predictedClass[i] in nested_list and targetClass[i] in nested_list:
                correctPhonemePosition.append(index)
                stateInPhoneme.append(i)
                #print("Both integers are in newArray[{index}]")
                #print("List is " + str(newArray[index]))

#print('corrrectphonemeindex: ', correctPhonemePosition)
#print('length og correct phoneme index: ', len(correctPhonemePosition))
'''

###Recalculating accuracy
correct = 0
for j in range(len(y_test)):
    if predictedClass[j] == targetClass[j]:
        correct +=1
print('correct: ', correct)

#print(stateInPhoneme)
#print('Amount of states in phoneme when the predicted and target is not the same: ', len(stateInPhoneme))

#correctPhonemes = len(stateInPhoneme)+correct
correctPhonemes = counter+correct
print('correct phonemes', correctPhonemes)

#Phoneme recognition:  
newAccuracy = 100 * (correctPhonemes/len(x_test))
print('Phoneme recognition accuracy: ', newAccuracy)

########### END PHONEME RECOGNITION



##################################### Semi-supervised learning: Teacher-student network ##################################### 

predicted = model.predict(x_train, batch_size=256, verbose=1)

model.save('model1.h5') 
del model


student = load_model('model1.h5')
student.summary()


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
history1 = student.fit(x=x_train, y=predicted, validation_data=(x_val, y_val),  batch_size=256, epochs=20, callbacks=[callback], verbose=1, shuffle=True)

score, acc = student.evaluate(x_test, y_test)
print('Test score: ', score)
print('Test accuracy: ', acc)

'''
fig, ax = plt.subplots()
plt.plot(history1.history['accuracy'], marker="*", label='Train accuracy', color='blue')
plt.plot(history1.history['val_accuracy'], marker="o", label='Test accuracy', color='red')
plt.title('Frame accuracy rate' )
plt.ylabel('Accuracy rate (%)')
plt.xlabel('Epoch')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=None, symbol=None))
#ax.grid()
plt.legend(loc='best')
plt.show()
plt.savefig('MONOSSL20.png')
'''


###################### PHONEME RECOGNITION #####################
predictedClass = student.predict_classes(x_test)  
#predictedClass = student.predict_classes(x_train)

stateInPhoneme = []
for i in range(len(y_test)):
    if (predictedClass[i] != targetClass[i]):
        for index, nested_list in enumerate(statesValues):
            if predictedClass[i] in nested_list and targetClass[i] in nested_list:
                stateInPhoneme.append(i)
                #print(index)

###Recalculating accuracy
correct = 0
for j in range(len(y_test)):
    #for j in range(len(x_train)):
    if predictedClass[j] == targetClass[j]:
        correct +=1
print('correct: ', correct)

#print(stateInPhoneme)
print('Amount of states in phoneme when the predicted and target is not the same: ', len(stateInPhoneme))

correctPhonemes = len(stateInPhoneme)+correct
print(correctPhonemes)

#Phoneme recognition:
newAccuracy = 100 * (correctPhonemes/len(predictedClass))
print('Phoneme recognition accuracy: ', newAccuracy)

########### END PHONEME RECOGNITION

posPhoneme = []
posTarget = []
for i in range(len(y_test)):
        for index, nested_list in enumerate(statesValues):
            if predictedClass[i] in nested_list:
                posPhoneme.append(index)
            if targetClass[i] in nested_list:
                posTarget.append(index)


y_pred = posPhoneme
y = posTarget


#Correctly predicted color CM
cm=confusion_matrix(y, y_pred)

#print(cm)
cm2 = cm+10**(-10)

'''
plt.figure(figsize=(31, 22))
#plt.figure(figsize=(27, 27))
plt.imshow(cm2, norm=colors.LogNorm(vmin=cm2.min(), vmax=cm2.max()))
tick_marks = np.arange(len(phonemes))
plt.xticks(tick_marks, phonemes, fontsize=24)
plt.yticks(tick_marks, phonemes, fontsize=24)
plt.tight_layout()
plt.savefig('cmgiampi22.png')
'''

'''
cm_df = pd.DataFrame(cm, index = phonemes, columns = phonemes)
cm_df2 = np.log(cm_df)
plt.figure(figsize=(11, 6))
plt.imshow(cm_df2, interpolation='nearest', aspect='auto', cmap='Greens')
tick_marks = np.arange(len(phonemes))
plt.xticks(tick_marks, phonemes, fontsize=9)
plt.yticks(tick_marks, phonemes, fontsize=9)
plt.savefig('CMmonomfcc.png')
'''

cm_df = pd.DataFrame(cm, index = phonemes, columns = phonemes)
cm_df2 = np.log(cm_df)
plt.figure(figsize=(10, 7))
plt.imshow(cm_df2, interpolation='nearest', aspect='auto', cmap='Greens')
tick_marks = np.arange(len(phonemes))
plt.xticks(tick_marks, phonemes, fontsize=8)
plt.yticks(tick_marks, phonemes, fontsize=8)
plt.savefig('CMmonomfcc.png')



