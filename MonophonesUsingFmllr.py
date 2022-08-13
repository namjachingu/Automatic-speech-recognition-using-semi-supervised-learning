from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


from kaldiio import ReadHelper
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import random
import sys
import itertools

#Show full numpy array
np.set_printoptions(threshold=sys.maxsize)

from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as mtick
from matplotlib import colors



train = dict()
with ReadHelper('scp:data-fmllr-tri3/train/feats.scp') as reader:
    for key, feats in reader:
        train[key] = feats


test = dict()
with ReadHelper('scp:data-fmllr-tri3/test/feats.scp') as reader:
    for key, feats in reader:
        test[key] = feats


dev = dict()
with ReadHelper('scp:data-fmllr-tri3/dev/feats.scp') as reader:
    for key, feats in reader:
        dev[key] = feats



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
#print(len(targets['mvjh0_si1556']))
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


phonemes = ['sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'cl', 'd', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'vcl', 'w', 'y', 'z', 'zh']


states = {'sil': [0, 1, 2], 'aa': [3, 4, 5], 'ae': [6, 7, 8], 'ah': [9, 10, 11], 'ao': [12, 13, 14], 'aw': [15, 16, 17],  'ax': [18, 19, 20], 'ay': [21, 22, 23], 'b': [24, 25, 26], 'ch': [27, 28, 29], 'cl': [30, 31, 32], 'd': [33, 34, 35], 'dh': [36, 37, 38], 'dx': [39, 40, 41], 'eh': [42, 43, 44], 'el': [45, 46, 47], 'en': [48, 49, 50], 'epi': [51, 52, 53], 'er': [54, 55, 56], 'ey': [57, 58, 59], 'f': [60, 61, 62], 'g': [63, 64, 65], 'hh': [66, 67, 68], 'ih': [69, 70, 71], 'ix': [72, 73, 74], 'iy': [75, 76, 77], 'jh': [78, 79, 80], 'k': [81, 82, 83], 'l': [84, 85, 86], 'm': [87, 88, 89], 'n': [90, 91, 92], 'ng': [93, 94, 95], 'ow': [96, 97, 98], 'oy': [99, 100, 101], 'p': [102, 103, 104], 'r': [105, 106, 107], 's': [108, 109, 110], 'sh': [111, 112, 113], 't': [114, 115, 116], 'th': [117, 118, 119], 'uh': [120, 121, 122], 'uw': [123, 124, 125], 'v': [126, 127, 128], 'vcl': [129, 130, 131], 'w': [132, 133, 134], 'y': [135, 136, 137], 'z': [138, 139, 140], 'zh': [141, 142, 143]}



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

model = Sequential()
model.add(Dense(440, activation='relu', input_shape=(440,)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dense(nTargets, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#40-dimensional fMLLR:
x_train = np.zeros((0, 440)) #40*11, 5 frames on each side of the current mfcc
x_test = np.zeros((0, 440))
x_dev = np.zeros((0, 440))
y_train = np.zeros((0, nTargets))
y_test = np.zeros((0, test_nTargets))
y_dev = np.zeros((0, dev_nTargets))

random.seed(0)
utterances = list(train.keys())
#random_keys=random.sample(utterances,3696)

#random_keys=random.sample(utterances, 37)
#random_keys=random.sample(utterances, 111)
#random_keys=random.sample(utterances, 185)
#random_keys=random.sample(utterances, 367)
#random_keys=random.sample(utterances, 739)
#random_keys=random.sample(utterances, 1109)
#print(len(random_keys))



#for keys in random_keys:
for keys in train.keys():
    fmllr = train[keys]
    x_mean = np.mean(fmllr, axis=0)
    x_std = np.std(fmllr, axis=0)
    train_normalized = ( fmllr - x_mean ) / x_std
    trainConc=frameConcat(train_normalized, 5, 1) #should give 13*11
    x_train = np.vstack((x_train, trainConc)) #concatenate mfcc

    targetsarray = targets[keys]
    #numberOfClasses = np.max(targetsarray)+1
    Labels = np.eye(nTargets)
    targetOneHot = Labels[targetsarray, :]
    y_train = np.vstack((y_train, targetOneHot)) #concatenated targets


for keys in dev.keys():
    devarray = dev[keys]
    dev_normalized = ( devarray - x_mean ) / x_std
    devConc=frameConcat(dev_normalized, 5, 1)
    x_dev = np.vstack((x_dev, devConc))

    targetsarray_dev = devTargets[keys]
    Labels_dev = np.eye(dev_nTargets)
    targetOneHot_dev = Labels_dev[targetsarray_dev, :]
    y_dev = np.vstack((y_dev, targetOneHot_dev))


for keys in test.keys():
    testarray = test[keys]
    test_normalized = ( testarray - x_mean ) / x_std
    testConc=frameConcat(test_normalized, 5, 1)
    x_test = np.vstack((x_test, testConc))

    targetsarray_test = testTargets[keys]
    Labels_test = np.eye(test_nTargets)
    targetOneHot_test = Labels_test[targetsarray_test, :]
    y_test = np.vstack((y_test, targetOneHot_test))


#Trainig with early stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
history = model.fit(x_train, y_train, validation_data=(x_dev, y_dev), batch_size=256, epochs=20, callbacks=[callback], verbose=1, shuffle=True)
numberOfEpochs = len(history.history['loss'])


score, acc = model.evaluate(x_test, y_test, batch_size=256, verbose=0)
print('Test score:', score)
print('Test accuracy:', acc)


fig, ax = plt.subplots()
plt.plot(history.history['accuracy'], marker="*", label='Train accuracy', color='blue')
plt.plot(history.history['val_accuracy'], marker="o", label='Test accuracy', color='red')
plt.title('Frame accuracy rate' )
plt.ylabel('Accuracy rate (%)')
plt.xlabel('Epoch')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=None, symbol=None))
plt.legend(loc='best')
plt.show()
plt.savefig('monofmllrSS.png')



############## PHONEME RECOGNITION ###############
targetClass = np.where(y_test==1)[1]
predictedClass = model.predict_classes(x_test) 


mapedState = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47] 


#MAKES LIST OF LIST FROM DICTIONARY
statesValues = []
for keys in states.keys():
    phonemeStates = states[keys] #this is a list
    #print(phonemeStates)
    statesValues.append(phonemeStates)


counter = 0
#stateInPhoneme = []
for i in range(len(y_test)):
    if (predictedClass[i] != targetClass[i]):
        for index, nested_list in enumerate(statesValues):
            if predictedClass[i] in nested_list and targetClass[i] in nested_list:
                #stateInPhoneme.append(i)
                counter += 1
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
print(correctPhonemes)

#Phoneme recognition:  
newAccuracy = 100 * (correctPhonemes/len(x_test))
print('Phoneme recognition accuracy: ', newAccuracy)

########### END PHONEME RECOGNITION




##################################### Semi-supervised learning: Student network #####################################

predicted = model.predict(x_train, batch_size=256, verbose=1)
#predicted = model.predict(x_train)

model.save('modelfmllr.h5')
del model

student = load_model('modelfmllr.h5')
student.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
history1 = student.fit(x=x_train, y=predicted, validation_data=(x_dev, y_dev),  batch_size=256, epochs=20, callbacks=[callback], verbose=1, shuffle=True)
#history1 = student.fit(x=predictionStudent, y=predictedclasses, validation_data=(x_val, y_val),  epochs=2)

score, acc = student.evaluate(x_test, y_test)
print('Test score: ', score)
print('Test accuracy: ', acc)

fig, ax = plt.subplots()
plt.plot(history1.history['accuracy'], marker="*", label='Train accuracy', color='blue')
plt.plot(history1.history['val_accuracy'], marker="o", label='Test accuracy', color='red')
plt.title('Frame accuracy rate' )
plt.ylabel('Accuracy rate (%)')
plt.xlabel('Epoch')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=None, symbol=None))
plt.legend(loc='best')
plt.show()
plt.savefig('monofmllrSSL.png')


########################### PHONEME RECOGNITION SSL ##########################
predictedClass = student.predict_classes(x_test)

stateInPhoneme = []
for i in range(len(y_test)):
    if (predictedClass[i] != targetClass[i]):
        for index, nested_list in enumerate(statesValues):
            if predictedClass[i] in nested_list and targetClass[i] in nested_list:
                stateInPhoneme.append(i)

###Recalculating accuracy
correct = 0
for j in range(len(y_test)):
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

#################### END PHONEME RECOGNITION SSL ########################



################### CONFUSION MATRIX ###################


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


plt.figure(figsize=(31, 22))
plt.imshow(cm2, norm=colors.LogNorm(vmin=cm2.min(), vmax=cm2.max()))
tick_marks = np.arange(len(phonemes))
plt.xticks(tick_marks, phonemes, fontsize=22)
plt.yticks(tick_marks, phonemes, fontsize=22)
plt.tight_layout()
plt.savefig('cmgiampiFMLLR.png')

'''
cm_df = pd.DataFrame(cm, index = phonemes, columns = phonemes)
cm_df2 = np.log(cm_df)
plt.figure(figsize=(31, 22))
plt.imshow(cm_df2, interpolation='nearest', aspect='auto', cmap='Greens')
tick_marks = np.arange(len(phonemes))
plt.xticks(tick_marks, phonemes, fontsize=17)
plt.yticks(tick_marks, phonemes, fontsize=17)
plt.colorbar()
plt.savefig('CMFMLLRSSL.png')
'''



cm_df = pd.DataFrame(cm, index = phonemes, columns = phonemes)
cm_df2 = np.log(cm_df)
plt.figure(figsize=(10, 7))
plt.imshow(cm_df2, interpolation='nearest', aspect='auto', cmap='Greens')
tick_marks = np.arange(len(phonemes))
plt.xticks(tick_marks, phonemes, fontsize=8)
plt.yticks(tick_marks, phonemes, fontsize=8)
plt.savefig('CMmonofmllr.png')



cm_df = pd.DataFrame(cm, index = phonemes, columns = phonemes)
cm_df2 = np.log(cm_df)
plt.figure(figsize=(10, 7))
plt.imshow(cm_df2, interpolation='nearest', aspect='auto', cmap='Blues')
tick_marks = np.arange(len(phonemes))
plt.xticks(tick_marks, phonemes, fontsize=8)
plt.yticks(tick_marks, phonemes, fontsize=8)
plt.savefig('CMmonofmllr2.png')

