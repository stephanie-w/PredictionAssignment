---
title: "Prediction Assignment"
author: "Stephanie W"
date: "21/8/2015"
output: html_document
---



## Synopsis

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

The data collected from accelerometers on the belt, forearm, arm, and dumbell are available at [Human Activity Recognition](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

The goal of this report is to use this dataset to predict which activity was performed at a specific point of time, ie. the values of the class variable (A, B, C, D or E) according to accelerometer measures recorded on the four different part of the body: arm, belt, dumbbell, forearm at this time.

## Getting data


```r
download.file(url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
              destfile="pml-training.csv", quiet=T, method="curl")
trainingData <- read.csv("pml-training.csv", na.strings=c("NA", "#DIV/0!"))
download.file(url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
              destfile="pml-testing.csv", quiet=T, method="curl")
validData <- read.csv("pml-testing.csv", na.strings=c("NA", "#DIV/0!"))
```



```r
table(trainingData$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```
The number of class A activities are much higher then for other class acitvities.

## Preprocessing the data

After examining the summary of the training set, we decide to keep with the "cleanest" variables (without NA's or unusual values), ie. the sensors measures:

```r
colnb <- grep("^(roll_|pitch_|yaw_|gyros_|total_accel|accel_|magnet_)", names(trainingData))
cols <- c("classe",names(trainingData)[colnb])
newTrainingData <- trainingData[, cols]
```


## Building a prediction model

Let's subset the data into a training and testing sets based on the `classe` variable (70% for training, 30% for testing):

```r
set.seed(1234)
library(caret)
inTrain<-createDataPartition(y=newTrainingData$classe, p=0.7, list=F)
training <- newTrainingData[inTrain,]
testing <- newTrainingData[-inTrain,]
```

We choose to use the Random Forest classification algorithm for its ability to handle large amounts of input variables.  
Let's run the algorithm on the training data with `classe` as outcome against all predictors:

```r
library(randomForest)
modelFit <- randomForest(training$classe~., data=training)
print(modelFit)
```

```
## 
## Call:
##  randomForest(formula = training$classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.51%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    3    0    0    1 0.001024066
## B   12 2643    3    0    0 0.005643341
## C    0   14 2380    2    0 0.006677796
## D    0    0   26 2225    1 0.011989343
## E    0    0    2    6 2517 0.003168317
```

We get an estimate of error rate $< 1\%$.

## Testing the model on the training set (testing)

Let's get the predictions from th testin set and compare the results with the actual data in a confusion matrix:

```r
pred <- predict(modelFit, testing)
confusionMatrix(pred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    8    0    0    0
##          B    0 1130    5    0    0
##          C    0    1 1021    4    0
##          D    0    0    0  959    1
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9966          
##                  95% CI : (0.9948, 0.9979)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9957          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9921   0.9951   0.9948   0.9991
## Specificity            0.9981   0.9989   0.9990   0.9998   0.9998
## Pos Pred Value         0.9952   0.9956   0.9951   0.9990   0.9991
## Neg Pred Value         1.0000   0.9981   0.9990   0.9990   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1920   0.1735   0.1630   0.1837
## Detection Prevalence   0.2858   0.1929   0.1743   0.1631   0.1839
## Balanced Accuracy      0.9991   0.9955   0.9970   0.9973   0.9994
```

We get a $\simeq 99\%$ of accuracy.


```r
table(pred,testing$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 1674    8    0    0    0
##    B    0 1130    5    0    0
##    C    0    1 1021    4    0
##    D    0    0    0  959    1
##    E    0    0    0    1 1081
```

## Testing the model on the validation set (validData)

Testing the model on the test set provided:

```r
pred <- predict(modelFit, validData)
pred
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
