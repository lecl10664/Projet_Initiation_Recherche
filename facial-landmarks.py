#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 11:13:32 2021

@author: leopoldclement
"""

# importer les paquets nécessaires
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2

# construire l'argument parser et analyser les arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
				help="chemin d'accès à l'image d'entrée")
args = vars(ap.parse_args())

# initialiser le détecteur de visage de dlib (basé sur HOG)
detector = dlib.get_frontal_face_detector()

# répertoire de modèles pré-formés
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# charger l'image d'entrée, redimensionner et convertir en niveaux de gris
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# détecter les visages
rects = detector(gray, 1)

# Pour chaque visage détecté, recherchez le repère.
for (i, rect) in enumerate(rects):
	# déterminer les repères du visage for the face region, then
	# convertir le repère du visage (x, y) en un array NumPy
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# convertir le rectangle de Dlib en un cadre de sélection de style OpenCV
	# dessiner le cadre de sélection
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
	# boucle sur les coordonnées (x, y) pour les repères faciaux
	# et dessine-les sur l'image
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        
# afficher l'image de sortie avec les détections de visage + repères de visage
cv2.imshow("Output", image)
cv2.waitKey(0)

