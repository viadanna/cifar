MLND Capstone Project
=====================
by Paulo Viadanna

This projects uses deep learning to identify objects in the CIFAR-10 dataset.
As such, it'll use the libraries scikit-learn, keras, tensorflow, pandas and numpy.
For plotting, matplotlib and seaborn will be used.
Check the requirements.txt file for further information.

Folder structure:
+ experiment.py is the main entry point that runs the models.
+ models.py contains each model implementation using Keras.
+ preprocessing.py is a helper to download, preprocess and augment the dataset.
+ inception\_v3.py contains the pre-trained Inception model, not used but kept for historical reasons.
+ inception\_hack.py contains the _tweaked_ Inception model that was used here.
+ run\_all.sh is a simple script to run the models as specified in the report.
+ results\_analysis.ipynb is a Jupyter notebook used to generate the plots.
+ helpers/ folder contains simple scripts to import training and results data.
+ input/ will contain the preprocessed datasets.
+ output/ will store the training history and results for each model.
