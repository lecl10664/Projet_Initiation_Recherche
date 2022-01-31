#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 12:19:24 2021

@author: leopoldclement
"""
# =============================================================================
# IMPORT
# =============================================================================
import cv2
import os
import dlib
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from seaborn import heatmap 
import matplotlib.pyplot as plt



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
data = {}

def get_landmarks(image):
    detections = detector(image, 1)
    landmarks = []
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
            landmarks.append(x)
            landmarks.append(y)
        data['landmarks'] = landmarks
    if len(detections) < 1:
        data['landmarks'] = "error"
    



def get_landmarks2(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
       
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            
        xmean = np.mean(xlist) #Find both coordinates of centre of gravity
        ymean = np.mean(ylist)
        
        xcentral = [(x-xmean) for x in xlist] #Calculate distance centre <-> other points in both axes
        ycentral = [(y-ymean) for y in ylist]
        
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"
        
    return data['landmarks_vectorised']
        



# frame = cv2.imread("images/test.jpg") 
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# clahe_image = clahe.apply(gray)
# get_landmarks(clahe_image)




# =============================================================================
# Classification 
# =============================================================================


# import dataset
data_path = '/Users/leopoldclement/Desktop/A2/Initiation_Recherche/CK+48'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        image=cv2.imread(data_path + '/'+ dataset + '/'+ img ) #open image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        get_landmarks(clahe_image)
        if data['landmarks'] == "error":
            print("no face detected on this one")
        else:
            img_data_list.append(data['landmarks']) #append image array to training data list
        
# prepare training and test      

labels = np.ones((927,),dtype='int64')
labels[0:206]=0 #207
labels[207:282]=1 #75
labels[282:531]=2 #249
labels[532:615]=3 #84
labels[616:750]=4 #135
labels[751:927]=5 #177

names = ['happy','fear','surprise','sadness','anger','disgust']
namesCourts = ['ha','fear','surp','sad','ang','dis']

def getLabel(id):
    return ['happy','fear','surprise','sadness','anger','disgust'][id]



#Shuffle the dataset
x,y = shuffle(img_data_list, labels, random_state=2)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=2)

#Turn the training set into a numpy array for the classifier
npar_train = np.array(X_train) 
npar_pred = np.array(X_test)


# =============================================================================
# Classificteur SVC
# =============================================================================

#train SVC
clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
clf.fit(npar_train, y_train)

pred_lin = clf.score(npar_pred, y_test)
print ("linear: ", pred_lin)
pred = clf.predict(npar_pred)

#Matrice de confusion


cm = confusion_matrix(y_test, pred, labels=[0,1,2,3,4,5], normalize='true')
print (cm)

from seaborn import heatmap 
import matplotlib.pyplot as plt
fig, ax= plt.subplots()
heatmap(cm, cmap="Blues", annot=True, ax = ax); #annot=True to annotate cells 

# labels, title and ticks 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix SVC');
ax.xaxis.set_ticklabels(names); ax.yaxis.set_ticklabels(namesCourts); 


#Test

frame = cv2.imread("images/peur.jpg") 
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_image = clahe.apply(gray)
get_landmarks(clahe_image)
test = data['landmarks']

clf.predict([test])


# =============================================================================
# Classificteur RandomForestClassifier
# =============================================================================

#train
clf = RandomForestClassifier(n_estimators=100)
clf.fit(npar_train, y_train)

pred_lin = clf.score(npar_pred, y_test)
print ("RandomForestClassifier: ", pred_lin)
pred = clf.predict(npar_pred)

#Matrice de confusion


cm = confusion_matrix(y_test, pred, labels=[0,1,2,3,4,5], normalize='true')
print (cm)

from seaborn import heatmap 
import matplotlib.pyplot as plt
fig, ax= plt.subplots()
heatmap(cm, cmap="Blues", annot=True, ax = ax); #annot=True to annotate cells 

# labels, title and ticks 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix RandomForestClassifier');
ax.xaxis.set_ticklabels(names); ax.yaxis.set_ticklabels(namesCourts); 


#Test

frame = cv2.imread("images/test3.png") 
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_image = clahe.apply(gray)
get_landmarks(clahe_image)
test = data['landmarks']

clf.predict([test])
 