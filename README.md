# Multi-classification on Žurnal24 dataset

Sample Usage:
```
$ python standalone_FINAL.py
```
This command will perform whole experiment (time consuming!!!):
 - download data and create dataset
 - preprocess data
 - perform learning on Logistic Regression and SGD
 - perform deep learning
 - create visualization
 - saves best model for prediction

```
$ python standalone_FINAL.py [file]
```
This command will predict class based on urls in provided file

