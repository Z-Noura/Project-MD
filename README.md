# Project MD
 
Code de reconstruction 3D à partir d'images 2D

Fichiers présents :
  * Génération d'images à partir de fichiers STL :
    - StlToMat.m : Conversion de fichiers STL en matrice 3D stockée dans un fichier .mat
    - MatToPngNpyParallel.py : Génération d'images à partir du fichier .mat en considérant une source envoyant des rayons parallèles
    - MatToPngNpyPinhole.py : Génération d'images à partir du fichier .mat en considérant une source ponctuelle (simule une caméra/radio réelle) dans des fichiers .png et .npy
  * Traitement d'images :
    - Traitement_Image.py : Génération d'images de l'objet filtré (segmentation) et des contours de l'objet (érosion, dilatation) dans des fichiers .png et .npy
    - FittingEllipse.py : Localisation des centres des cercles (ellipses) dans une image et stockage de ces données dans un fichier .npy
  * Reconstruction 3D :
    - Stereovision : Méthode stéréovision (géométrie épipolaire)
    - Visual Hull : Méthode Visual Hull
