# SP-Final-Project
The code and data for SP Final Project. This code takes data from the Wada app and uses classifiers from Weka.jar.

## Specific Code Information for Final Project
All data processing tasks, including creating time slices and merging files to generate datasets for the left wrist and the combined left and right wrist data, are organized within **"final_project.ipynb"**.

The code used to find the best time slices and evaluate each classifier with all the features is located in **"Assignment2part3.java"**.

The code used to perform Sequential Feature Selection on each classifier is located in **"Assignment2part4.java"**.

## Data
Those csv files under the same folder with name **"combined"** in them are what you need. I realized that this version of github can be too crowded, so this is just our first version. I can upload a clean version in the near future.

## Data Processing
All the process for data processing is in the **"final_project.ipynb"**

## Classification
I used Assignment 2 part2 and part5 to provide inspiration. Then **"ToothbrushClassification.java"** is my first version of this project. I changed **"MyWekaUtils.java"** and add two more classifier in it.

Based on that I used **"feature_selection.java"** and **"vote.java"** as second attempt in simple classifier method.

Finally, I printed the parameters of NeuralNetwork in **"ToothbrushClassification.java"**. I build up my own NN in **"NN.java"**.
