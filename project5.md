Project
Here I used PCA and random forest plot

#Loading required libraries

# Loading the data

```r
if (!file.exists("pml-training.csv")) {
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",method = "auto",destfile = "pml-training.csv")
}
if (!file.exists("pml-testing.csv")) {
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",method = "auto",destfile = "pml-testing.csv")
}
training<-read.csv("pml-training.csv",stringsAsFactors=F,na.strings=c("","#DIV/0!","NA"))
testing<-read.csv("pml-testing.csv",stringsAsFactors=F,na.strings=c("","#DIV/0!","NA"))
```

# Preprocessing of the data file

## Cleaning of the data file
I cleaned the data file by removing all irrelevant and useless predictors
### Rememoving irrelevant variables
first I removed all the variables that were unrelated to the prediction. i.e., the first seven variables

```r
training<-training[,8:160]
testing<-testing[,8:160]
```


### Removing useless varibles
Then, I removed all variables with zero variance as they will of no value in prediction

```r
nsv<-nearZeroVar(training)
nsv
```

```
##  [1]   7  10  19  44  45  46  47  48  49  50  51  52  68  71  72  74  75
## [18]  82  85  94 120 123 124 127 130 132 135 136 137 138 139 140 141 142
## [35] 143
```

```r
training<-(training[-nsv])
testing<-(testing[-nsv])
dim(training);dim(testing)
```

```
## [1] 19622   118
```

```
## [1]  20 118
```
## Pre-processing of the data file
Before pre-processing of the data, I have to remove the outcome varialbe-after saving it in an object named outcome-and convert the variables into numeric variables

```r
#save the training$class into outcome 
outcome<-training$classe
#Remove the outcome variable
training$classe<-NULL
#Convert all variable into numeric
new<-sapply(training,as.numeric)
#convert the dataframe into notcentered
notcentered<-as.data.frame(new)
```
I imputed missing values using knn method

```r
preProc3<-preProcess(notcentered,method="knnImpute")
dim(notcentered)
```

```
## [1] 19622   117
```

```r
imputed<-predict(preProc3,notcentered)
dim(notcentered);dim(imputed)
```

```
## [1] 19622   117
```

```
## [1] 19622   117
```
# Fitting of a randomforest model
Before I applied the model, I splited traing into train and test

```r
imputed$classe<-outcome
set.seed(123)
inTrain <- createDataPartition(y=imputed$classe,p=0.75, list=FALSE)
train <- imputed[inTrain,]
test <- imputed[-inTrain,]
rbind("original dataset" = dim(imputed),
      "training set" = dim(train),
      "testing set"=dim(test))
```

```
##                   [,1] [,2]
## original dataset 19622  118
## training set     14718  118
## testing set       4904  118
```

```r
table(train$classe)
```

```
## 
##    A    B    C    D    E 
## 4185 2848 2567 2412 2706
```

```r
table(test$classe)
```

```
## 
##    A    B    C    D    E 
## 1395  949  855  804  901
```



#Exploratory data analysis


I trained the random forest model.

```r
#Train a randomForest model
dim(train);dim(test);names(train)
```

```
## [1] 14718   118
```

```
## [1] 4904  118
```

```
##   [1] "roll_belt"                "pitch_belt"              
##   [3] "yaw_belt"                 "total_accel_belt"        
##   [5] "kurtosis_roll_belt"       "kurtosis_picth_belt"     
##   [7] "skewness_roll_belt"       "skewness_roll_belt.1"    
##   [9] "max_roll_belt"            "max_picth_belt"          
##  [11] "max_yaw_belt"             "min_roll_belt"           
##  [13] "min_pitch_belt"           "min_yaw_belt"            
##  [15] "amplitude_roll_belt"      "amplitude_pitch_belt"    
##  [17] "var_total_accel_belt"     "avg_roll_belt"           
##  [19] "stddev_roll_belt"         "var_roll_belt"           
##  [21] "avg_pitch_belt"           "stddev_pitch_belt"       
##  [23] "var_pitch_belt"           "avg_yaw_belt"            
##  [25] "stddev_yaw_belt"          "var_yaw_belt"            
##  [27] "gyros_belt_x"             "gyros_belt_y"            
##  [29] "gyros_belt_z"             "accel_belt_x"            
##  [31] "accel_belt_y"             "accel_belt_z"            
##  [33] "magnet_belt_x"            "magnet_belt_y"           
##  [35] "magnet_belt_z"            "roll_arm"                
##  [37] "pitch_arm"                "yaw_arm"                 
##  [39] "total_accel_arm"          "var_accel_arm"           
##  [41] "gyros_arm_x"              "gyros_arm_y"             
##  [43] "gyros_arm_z"              "accel_arm_x"             
##  [45] "accel_arm_y"              "accel_arm_z"             
##  [47] "magnet_arm_x"             "magnet_arm_y"            
##  [49] "magnet_arm_z"             "kurtosis_roll_arm"       
##  [51] "kurtosis_picth_arm"       "kurtosis_yaw_arm"        
##  [53] "skewness_roll_arm"        "skewness_pitch_arm"      
##  [55] "skewness_yaw_arm"         "max_picth_arm"           
##  [57] "max_yaw_arm"              "min_yaw_arm"             
##  [59] "amplitude_yaw_arm"        "roll_dumbbell"           
##  [61] "pitch_dumbbell"           "yaw_dumbbell"            
##  [63] "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
##  [65] "skewness_roll_dumbbell"   "skewness_pitch_dumbbell" 
##  [67] "max_roll_dumbbell"        "max_picth_dumbbell"      
##  [69] "max_yaw_dumbbell"         "min_roll_dumbbell"       
##  [71] "min_pitch_dumbbell"       "min_yaw_dumbbell"        
##  [73] "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
##  [75] "total_accel_dumbbell"     "var_accel_dumbbell"      
##  [77] "avg_roll_dumbbell"        "stddev_roll_dumbbell"    
##  [79] "var_roll_dumbbell"        "avg_pitch_dumbbell"      
##  [81] "stddev_pitch_dumbbell"    "var_pitch_dumbbell"      
##  [83] "avg_yaw_dumbbell"         "stddev_yaw_dumbbell"     
##  [85] "var_yaw_dumbbell"         "gyros_dumbbell_x"        
##  [87] "gyros_dumbbell_y"         "gyros_dumbbell_z"        
##  [89] "accel_dumbbell_x"         "accel_dumbbell_y"        
##  [91] "accel_dumbbell_z"         "magnet_dumbbell_x"       
##  [93] "magnet_dumbbell_y"        "magnet_dumbbell_z"       
##  [95] "roll_forearm"             "pitch_forearm"           
##  [97] "yaw_forearm"              "kurtosis_roll_forearm"   
##  [99] "kurtosis_picth_forearm"   "skewness_roll_forearm"   
## [101] "skewness_pitch_forearm"   "max_picth_forearm"       
## [103] "max_yaw_forearm"          "min_pitch_forearm"       
## [105] "min_yaw_forearm"          "amplitude_pitch_forearm" 
## [107] "total_accel_forearm"      "var_accel_forearm"       
## [109] "gyros_forearm_x"          "gyros_forearm_y"         
## [111] "gyros_forearm_z"          "accel_forearm_x"         
## [113] "accel_forearm_y"          "accel_forearm_z"         
## [115] "magnet_forearm_x"         "magnet_forearm_y"        
## [117] "magnet_forearm_z"         "classe"
```

```r
y<-train$classe
y<-as.factor(y)
x<-train[,1:117]
modFit <- randomForest(x,y,
                importance=T,ntree=200, nodesize=25)
```
I determine the important variables by arrangind them according to the number of trees that use them
The following plot shows the predictors arranged according to their importance in prediction

```r
##PLot used variables
vu<-varUsed(modFit,count=T)
vusorted = sort(vu, decreasing = F, index.return = TRUE)
vusorteddes = sort(vu, decreasing = T, index.return = TRUE)
dotchart(vusorted$x, names(modFit$forest$xlevels[vusorted$ix]))
```

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9-1.png) 
I save the indices of the important pedictors in a vector named imp. I used to select the important predictors in the new model

```r
imp<-vusorteddes$ix[1:30]
imp
```

```
##  [1]   3   1   2  94  93  92  95  96  90  35  60 117  91  33  87  62  36
## [18]  29  32  34 112 114 116  75  48  37 110  38 115  49
```

```r
x2<-train[,imp]
dim(x2)
```

```
## [1] 14718    30
```

```r
modFit2 <- randomForest(x2,y,
                       importance=T,ntree=200, nodesize=25)
```
To evaluate the perfromance of the two models, the redudnant model- the model with all the predictor- and the concise model- the model containing the most important 30 predictors only. The performance of the concise model is a little higher than that of the redundant model althought it uses only 30 predictors.Thus, we will use the concise model
#Predict the performance of the redundant model
predictions=predict(modFit,test)
#Predict the performance of the consice model
predictions2=predict(modFit2,test)
rbind("Accuracy of the redundant model"=confusionMatrix(test$classe,predictions)$overall[1], "Accuracy of the concise model"=confusionMatrix(test$classe,predictions2)$overall[1])


# Predict the test set
First, we have to prepare the testing set to be used for prediction. We have to convert it into numeric variables before imputing the missing values


```r
testing$classe<-NULL
newTest<-sapply(testing,as.numeric)
notcenteredTest<-as.data.frame(newTest)
imputedTest<-predict(preProc3,notcenteredTest[,1:117])
```
I used the concise model to predict the outcome variable in the test set


```r
predictTest2<-predict(modFit2,imputedTest)
predictTest2
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

I used the following function to save the differet predicted values into 20 text file for submission

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(predictTest2)
```
