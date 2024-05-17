# Project MD
 
Code de reconstruction 3D à partir d'images 2D


 Fichiers presents
      - Generation images à partir de fichiers STL:
           StlToMat.m : Conversion fichier stl en matrice 3D stockée en fichier .mat
           MatToPngNpyParrallel.py : Generation d'images à partir du ficher .mat en considerant une source envoyant des rayons paralleles
           MatToPngNpyPinhole.py : Generation d'images à partir du ficher .mat en considerant une source ponctuelle (simule une camera/radio reele) dans un fichier .png et .np       
       - Traitement d'images: 
           Tratement_Image.py : Generation d'imes de l'objet filtrée (segmentation) et les contours de l'objet (erosion,dilatation) dans un fichier .png et .npy
           FittingEllipse.py : Localisationdes centres des cercles (ellipses) dans une image et socke ces données dans un fichier.npy
       - Reconstruciton 3D:
           Stereovision : methode stereovision (Geometrie epipolaire)
           Visual Hull : Methode Visual Hull
