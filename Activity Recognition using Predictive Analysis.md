Overview

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit, it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. The aim of this project is to predict the
manner in which participants perform a barbell lift. The data comes from
<a href="http://groupware.les.inf.puc-rio.br/har" class="uri">http://groupware.les.inf.puc-rio.br/har</a>
wherein 6 participants were asked to perform the same set of exercises
correctly and incorrectly with accelerometers placed on the belt,
forearm, arm, and dumbell.

For the purpose of this project, the following steps would be followed:

1.  Data Preprocessing
2.  Exploratory Analysis
3.  Prediction Model Selection
4.  Predicting Test Set Output
5.  Data Preprocessing

First, we load the training and testing set from the online sources and
then split the training set further into training and test sets.

    library(caret)

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    trainURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    testURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

    training <- read.csv(url(trainURL))
    testing <- read.csv(url(testURL))

    label <- createDataPartition(training$classe, p = 0.7, list = FALSE)
    train <- training[label, ]
    test <- training[-label, ]

From among 160 variables present in the dataset, some variables have
nearly zero variance whereas some contain a lot of NA terms which need
to be excluded from the dataset. Moreover, other 5 variables used for
identification can also be removed.

    NZV <- nearZeroVar(train)
    train <- train[ ,-NZV]
    test <- test[ ,-NZV]

    label <- apply(train, 2, function(x) mean(is.na(x))) > 0.95
    train <- train[, -which(label, label == FALSE)]
    test <- test[, -which(label, label == FALSE)]

    train <- train[ , -(1:5)]
    test <- test[ , -(1:5)]

As a result of the preprocessing steps, we were able to reduce 160
variables to 54.

Exploratory Analysis

Now that we have cleaned the dataset off absolutely useless varibles, we
shall look at the dependence of these variables on each other through a
correlation plot.

    library(corrplot)

    ## corrplot 0.84 loaded

    corrMat <- cor(train[,-54])
    corrplot(corrMat, method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0,0,0))

![](Predictive-Analysis_files/figure-markdown_strict/unnamed-chunk-3-1.png)

In the plot above, darker gradient correspond to having high
correlation. A Principal Component Analysis can be run to further reduce
the correlated variables but we aren’t doing that due to the number of
correlations being quite few.

Prediction Model Selection

We will use 3 methods to model the training set and thereby choose the
one having the best accuracy to predict the outcome variable in the
testing set. The methods are Decision Tree, Random Forest and
Generalized Boosted Model.

A confusion matrix plotted at the end of each model will help visualize
the analysis better.

Decision Tree

    library(rpart)
    library(rpart.plot)
    library(rattle)

    ## Loading required package: tibble

    ## Loading required package: bitops

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.4.0 Copyright (c) 2006-2020 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

    set.seed(13908)
    modelDT <- rpart(classe ~ ., data = train, method = "class")
    fancyRpartPlot(modelDT)

![](Predictive-Analysis_files/figure-markdown_strict/unnamed-chunk-4-1.png)

    predictDT <- predict(modelDT, test, type = "class")
    confMatDT <- confusionMatrix(predictDT, test$classe)
    confMatDT

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1404   75    2   14    5
    ##          B   89  793   38   36   84
    ##          C   46  118  837   75   23
    ##          D  128  137  131  782  189
    ##          E    7   16   18   57  781
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7811          
    ##                  95% CI : (0.7704, 0.7916)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.7248          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8387   0.6962   0.8158   0.8112   0.7218
    ## Specificity            0.9772   0.9480   0.9461   0.8811   0.9796
    ## Pos Pred Value         0.9360   0.7625   0.7616   0.5721   0.8885
    ## Neg Pred Value         0.9384   0.9286   0.9605   0.9597   0.9399
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2386   0.1347   0.1422   0.1329   0.1327
    ## Detection Prevalence   0.2549   0.1767   0.1867   0.2323   0.1494
    ## Balanced Accuracy      0.9080   0.8221   0.8809   0.8462   0.8507

Random Forest

    library(caret)
    set.seed(13908)
    control <- trainControl(method = "cv", number = 3, verboseIter=FALSE)
    modelRF <- train(classe ~ ., data = train, method = "rf", trControl = control)
    modelRF$finalModel

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.15%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3905    1    0    0    0 0.0002560164
    ## B    3 2653    2    0    0 0.0018811136
    ## C    0    3 2393    0    0 0.0012520868
    ## D    0    0    5 2247    0 0.0022202487
    ## E    0    1    0    6 2518 0.0027722772

    predictRF <- predict(modelRF, test)
    confMatRF <- confusionMatrix(predictRF, test$classe)
    confMatRF

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    1    0    0    0
    ##          B    0 1137    3    0    2
    ##          C    0    1 1023    5    0
    ##          D    0    0    0  959    5
    ##          E    1    0    0    0 1075
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9969          
    ##                  95% CI : (0.9952, 0.9982)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9961          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9982   0.9971   0.9948   0.9935
    ## Specificity            0.9998   0.9989   0.9988   0.9990   0.9998
    ## Pos Pred Value         0.9994   0.9956   0.9942   0.9948   0.9991
    ## Neg Pred Value         0.9998   0.9996   0.9994   0.9990   0.9985
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2843   0.1932   0.1738   0.1630   0.1827
    ## Detection Prevalence   0.2845   0.1941   0.1749   0.1638   0.1828
    ## Balanced Accuracy      0.9996   0.9986   0.9979   0.9969   0.9967

Generalized Boosted Model

    library(caret)
    set.seed(13908)
    control <- trainControl(method = "repeatedcv", number = 5, repeats = 1, verboseIter = FALSE)
    modelGBM <- train(classe ~ ., data = train, trControl = control, method = "gbm", verbose = FALSE)
    modelGBM$finalModel

    ## A gradient boosted model with multinomial loss function.
    ## 150 iterations were performed.
    ## There were 53 predictors of which 53 had non-zero influence.

    predictGBM <- predict(modelGBM, test)
    confMatGBM <- confusionMatrix(predictGBM, test$classe)
    confMatGBM

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1668    5    0    0    0
    ##          B    5 1127   18    5    5
    ##          C    0    6 1004    9    1
    ##          D    1    0    4  949   15
    ##          E    0    1    0    1 1061
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9871          
    ##                  95% CI : (0.9839, 0.9898)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9837          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9964   0.9895   0.9786   0.9844   0.9806
    ## Specificity            0.9988   0.9930   0.9967   0.9959   0.9996
    ## Pos Pred Value         0.9970   0.9716   0.9843   0.9794   0.9981
    ## Neg Pred Value         0.9986   0.9975   0.9955   0.9969   0.9956
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2834   0.1915   0.1706   0.1613   0.1803
    ## Detection Prevalence   0.2843   0.1971   0.1733   0.1647   0.1806
    ## Balanced Accuracy      0.9976   0.9913   0.9876   0.9902   0.9901

As Random Forest offers the maximum accuracy of 99.78%, we will go with
Random Forest Model to predict our test data class variable.

Predicting Test Set Output

    predictRF <- predict(modelRF, testing)
    predictRF

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
