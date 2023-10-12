# Proyecto-Modelado-Predictivo-Magic

This repository contains the code for the analysis and modeling of the magic04 dataset.

## Model Types

There are two main types of models available for the magic04 dataset:

1. **Linear Models**: This type of model focuses on using linear regression techniques for the analysis and modeling of the magic04 dataset.
2. **SVC Models**: This type of model utilizes Support Vector Classifier (SVC) techniques for the analysis and modeling of the magic04 dataset.

Both types of models were trained using either standarization or multiple transformation techniques. The models were then saved as pickle files for later use. 

## Report

The `Report` directory contains the final report for the project.

## To_Test

The `To_Test` directory contains the necessary files to duplicate the models using the pickle files. It includes the following files:

- `MAGIC04_1V5.ipynb`: This Jupyter notebook contains the code to test the logistic regression models using the pickle files and magic04 dataset.
- `svc.py`: This Python script contains the code to test the SVC models using the pickle files.

## Pickles

The `pickles` directory contains the pickle files that can be used to reproduce the examples.

## Optimizacion_hiperparameters

The `Optimizacion_hiperparameters` directory contains the code to optimize the hyperparameters of the SVC models.

## MAGIC04.data

The `MAGIC04.data` file contains the dataset. Its columns can be obtained from MAGIC04.names