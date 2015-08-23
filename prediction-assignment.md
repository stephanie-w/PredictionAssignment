---
title: "Prediction Assignment"
author: "Stephanie W"
date: "21/8/2015"
output: html_document
---



## Synopsis

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

The data collected from accelerometers on the belt, forearm, arm, and dumbbell are available at [Human Activity Recognition](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

The goal of this report is to use this dataset to predict which activity was performed at a specific point of time, ie. the values of the class variable (A, B, C, D or E) according to accelerometer measures recorded on the four different part of the body: arm, belt, dumbbell, forearm at this time.

## Getting the data


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

The number of class A activities are much higher then for other class activities.

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
modelFit <- randomForest(training$classe~., data=training, importance = TRUE)
print(modelFit)
```

```
## 
## Call:
##  randomForest(formula = training$classe ~ ., data = training,      importance = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.47%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3903    2    0    0    1 0.0007680492
## B   12 2641    5    0    0 0.0063957863
## C    0   11 2383    2    0 0.0054257095
## D    0    0   23 2227    2 0.0111012433
## E    0    0    2    5 2518 0.0027722772
```

We get an estimate of error rate $< 1\%$.

Let’s look at what variables were important:

```r
varImp(modelFit)
```

```
##                             A        B        C        D        E
## roll_belt            36.08585 44.06706 43.98599 45.11099 40.85850
## pitch_belt           30.07637 49.42134 39.79201 33.79727 30.44938
## yaw_belt             47.03527 42.76429 41.87899 46.75578 33.83133
## total_accel_belt     13.60783 16.32316 14.66852 13.92725 16.20039
## gyros_belt_x         17.98072 17.03353 21.50826 13.16885 15.39356
## gyros_belt_y         11.70681 16.34217 16.24090 13.45163 17.77025
## gyros_belt_z         22.57380 28.40942 23.66059 21.80462 25.77383
## accel_belt_x         14.15274 16.93674 16.81779 14.99527 15.13749
## accel_belt_y         11.99300 13.71650 14.37281 14.65105 13.59182
## accel_belt_z         20.07750 22.24683 24.30927 21.53925 20.43154
## magnet_belt_x        17.30526 25.34803 25.19573 19.94333 24.43781
## magnet_belt_y        21.36812 26.30906 25.37254 26.26617 23.19595
## magnet_belt_z        22.71844 28.37315 23.83690 29.93210 25.37440
## roll_arm             19.88401 29.79294 26.74108 28.84946 21.28468
## pitch_arm            17.85796 24.54099 20.58568 20.73095 18.59464
## yaw_arm              25.16739 26.36315 25.49231 28.49305 20.52944
## total_accel_arm       9.53086 23.40861 21.15082 19.56435 19.74600
## gyros_arm_x          17.14964 26.93674 22.63004 25.25380 19.41033
## gyros_arm_y          19.80134 27.66242 22.12970 26.05521 20.62971
## gyros_arm_z          13.88414 16.74068 15.57302 13.81887 14.05196
## accel_arm_x          17.38027 19.47605 20.36039 22.71360 16.75619
## accel_arm_y          18.53653 24.13101 18.20714 20.24059 15.83599
## accel_arm_z          12.55949 19.41027 20.47467 21.70112 16.83599
## magnet_arm_x         15.50978 14.74534 17.64661 17.08726 14.46626
## magnet_arm_y         13.05258 16.98432 19.35154 21.56999 13.64057
## magnet_arm_z         21.39693 24.51493 23.17218 21.90631 20.06110
## roll_dumbbell        23.58023 25.85482 28.68305 27.93218 25.52478
## pitch_dumbbell       12.43285 19.07278 16.87023 12.60673 14.65477
## yaw_dumbbell         17.76052 24.98185 24.72322 22.38195 23.76837
## total_accel_dumbbell 19.33487 25.03121 20.67877 22.81858 23.04000
## gyros_dumbbell_x     16.93342 24.43346 22.12978 19.21593 18.55923
## gyros_dumbbell_y     22.91353 22.33652 26.76344 22.23393 18.35558
## gyros_dumbbell_z     19.91661 24.75339 19.14614 18.18368 16.82908
## accel_dumbbell_x     16.30983 23.77734 20.94856 18.99903 20.09600
## accel_dumbbell_y     26.69030 29.19876 31.54304 27.93606 27.32845
## accel_dumbbell_z     22.56468 28.99258 27.09775 27.67768 29.78778
## magnet_dumbbell_x    25.58071 25.62644 28.81375 26.50648 23.50184
## magnet_dumbbell_y    34.73483 35.67495 43.83125 35.16517 31.34239
## magnet_dumbbell_z    43.81781 39.73025 48.17504 39.49114 37.11960
## roll_forearm         28.04641 24.09614 31.44878 23.23632 23.17462
## pitch_forearm        31.07832 31.91974 34.59602 34.17348 32.21701
## yaw_forearm          18.29877 21.19990 20.66878 20.73287 20.54302
## total_accel_forearm  18.91253 19.16081 21.35473 17.56886 18.56895
## gyros_forearm_x      13.44276 17.49002 19.92761 15.56288 15.66800
## gyros_forearm_y      16.83177 28.53653 27.16373 23.60499 19.36001
## gyros_forearm_z      16.47271 24.89179 22.23111 18.80051 16.59016
## accel_forearm_x      17.98762 25.40509 25.65041 28.85027 22.02177
## accel_forearm_y      19.24324 22.24429 20.73216 17.93000 20.29311
## accel_forearm_z      18.69581 23.05065 24.42607 22.13416 22.80749
## magnet_forearm_x     16.11134 20.57865 20.76289 18.51767 20.11752
## magnet_forearm_y     21.45605 23.22177 24.43055 22.70157 21.84645
## magnet_forearm_z     25.50577 27.19605 27.52183 28.73788 26.34341
```

## Testing the model on the training set (testing)

Let's get the predictions from the testing set and compare the results with the actual data in a confusion matrix:

```r
pred <- predict(modelFit, testing)
confusionMatrix(pred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    6    0    0    0
##          B    0 1132    6    0    0
##          C    0    1 1020    4    0
##          D    0    0    0  959    0
##          E    0    0    0    1 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9969          
##                  95% CI : (0.9952, 0.9982)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9961          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9939   0.9942   0.9948   1.0000
## Specificity            0.9986   0.9987   0.9990   1.0000   0.9998
## Pos Pred Value         0.9964   0.9947   0.9951   1.0000   0.9991
## Neg Pred Value         1.0000   0.9985   0.9988   0.9990   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1924   0.1733   0.1630   0.1839
## Detection Prevalence   0.2855   0.1934   0.1742   0.1630   0.1840
## Balanced Accuracy      0.9993   0.9963   0.9966   0.9974   0.9999
```

We get a $\simeq 99\%$ of accuracy.


```r
table(pred,testing$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 1674    6    0    0    0
##    B    0 1132    6    0    0
##    C    0    1 1020    4    0
##    D    0    0    0  959    0
##    E    0    0    0    1 1082
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
