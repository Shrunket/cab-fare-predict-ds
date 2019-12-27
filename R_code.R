rm(list=ls())
getwd()
setwd('C:/Users/Samruddhi/Desktop/Edwisor Project 1')

train_data= read.csv('train_cab.csv')
test_data= read.csv('test.csv')
head(train_data)
head(test_data)
str(train_data)
str(test_data)
train_data$fare_amount= as.numeric(as.character(train_data$fare_amount))

num_col= c('fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count')
for (i in num_col){
 print(summary(train_data[i]))
}
test_num_col= c('pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count')
for (i in test_num_col){
  print(summary(test_data[i]))
}
hist(train_data$passenger_count, main='Histogram of passenger_count')

################################### Missing Value Analysis###################################################
train_data[111,1] #9
train_data[569,1] #6.5
train_data[11000,1] #9.7
train_data[3,1] #5.7
train_data[6038,1] #6
#train_data$fare_amount[is.na(train_data$fare_amount)]= mean(train_data$fare_amount,na.rm = TRUE) # 15.02
#train_data$fare_amount[is.na(train_data$fare_amount)]= median(train_data$fare_amount,na.rm = TRUE) # 8.50
library(dplyr)
library(DMwR)

train_data_1= knnImputation(train_data,k=5) #Value of KNN with k=5 is more near to the actual values as than to median
train_data_2= knnImputation(train_data,k=5,meth='median')
train_data_1$passenger_count=train_data_2$passenger_count
str(train_data_1)
train_data= train_data_1

summary(train_data$fare_amount)
summary(train_data$passenger_count)
summary(train_data)

############################################# Outlier Analysis ###############################################
boxplot(train_data$passenger_count, main='Boxplot of passenger_count')

boxplot.stats(train_data$fare_amount)
barplot(table(train_data$fare_amount),ylim = c(0,100))
train_data= train_data[train_data$fare_amount > 0 & train_data$fare_amount < 100, ]
train_data= train_data[train_data$passenger_count < 8 & train_data$passenger_count >=1,]

summary(train_data)
boxplot(test_data$passenger_count)

num_col= c('fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude')
for(i in num_col){
  print(i)
  val = train_data[,i][train_data[,i] %in% boxplot.stats(train_data[,i])$out]
  print(length(val))
  train_data = train_data[which(!train_data[,i] %in% val),]
}

################################################################################################################

############################################# Feature Engineering #############################################
train_data=train_data[!train_data$pickup_datetime==43,]
library(geosphere)
distance= function(pickup_long,pickup_lat,dropoff_long,dropoff_lat){
  trip=distHaversine(c(pickup_long,pickup_lat),c(dropoff_long,dropoff_lat))
  return(trip)
}
for (i in 1:nrow(train_data)){
  attempt=distance(train_data$pickup_longitude[i],train_data$pickup_latitude[i],train_data$dropoff_longitude[i],train_data$dropoff_latitude[i])
  train_data$trip_distance[i]= attempt/1609.344
}
for (i in 1:nrow(test_data)){
  attempt=distance(test_data$pickup_longitude[i],test_data$pickup_latitude[i],test_data$dropoff_longitude[i],test_data$dropoff_latitude[i])
  test_data$trip_distance[i]= attempt/1609.344
}

summary(train_data$trip_distance)


library(lubridate)
for (i in 1:nrow(train_data)){
  train_data$year[i]= year(train_data$pickup_datetime[i])
  train_data$month[i]= month(train_data$pickup_datetime[i])
  train_data$day[i]= day(train_data$pickup_datetime[i])
  train_data$hour[i]= hour(train_data$pickup_datetime[i])
  train_data$minute[i]= minute(train_data$pickup_datetime[i])
  train_data$second[i]= second(train_data$pickup_datetime[i])
}
for (i in 1:nrow(test_data)){
  test_data$year[i]= year(test_data$pickup_datetime[i])
  test_data$month[i]= month(test_data$pickup_datetime[i])
  test_data$day[i]= day(test_data$pickup_datetime[i])
  test_data$hour[i]= hour(test_data$pickup_datetime[i])
  test_data$minute[i]= minute(test_data$pickup_datetime[i])
  test_data$second[i]= second(test_data$pickup_datetime[i])
}

train_data=train_data[,-c(2)]
test_data= test_data[,-c(1)]
train_data= train_data[!train_data$trip_distance==0,]
#write.csv(train_data,'final_train.csv',row.names = F)


########################################### Feature Selection #################################################
library(corrplot)
con_var= c('fare_amount','pickup_longitude','pickup_latitude',"dropoff_longitude","dropoff_latitude","trip_distance","year","month","day","hour","minute","second")
corrplot(cor(train_data),method='pie')
cor(train_data)
train_data= train_data[,-c(2,4)]

########################################### Feature Scaling #####################################################
hist(train_data$passenger_count)
boxplot(train_data$passenger_count)
std_var=c('pickup_longitude','pickup_latitude',"dropoff_longitude","dropoff_latitude")
norm_col=c('trip_distance','year','month','day','hour','minute','second')
# Normalisaztion on norm_col
for(i in norm_col){
  print(i)
  train_data[,i] = (train_data[,i] - min(train_data[,i]))/
    (max(train_data[,i] - min(train_data[,i])))
}
#Standardisation on std_var
for(i in std_var){
  print(i)
  train_data[,i] = (train_data[,i] - mean(train_data[,i]))/
    (sd(train_data[,i] ))
}
corrgram(train_data)
########################################### Model Development##################################################


# Data Split using cross validation / k-fold cross validation
library(tidyverse)
library(caret)
library(party)
set.seed(123)

train_con= trainControl(method = 'oob',number = 10)

# Multiple Algorithms
LR_model= train(fare_amount~.,data = train_data,method='rf',trControl= train_con)
LR_model
# Linear Regression: RMSE: 2.0925 Rsquared:0.7015  MAE: 1.4996
# Logistic Regression: RMSE: 2.0925 Rsquared:0.7015  MAE: 1.4996
# Decision Tree: RMSE: 2.0788 Rsquared:0.70539  MAE: 1.4822
# KNN Algorithm : RMSE: 3.6384 Rsquared:0.1022  MAE: 2.8172
# Random Forest: RMSE: 1.9454 Rsquared:0.7419  mtry:7 MAE: 
# Here we can proceed with model building using Random Forest

#Hyper-parameter tuning

# choosing optimum value for mtry
set.seed(1234)
tune_Grid = expand.grid(.mtry = c(1: 10))
rf_mtry = train(fare_amount~.,
                data = train_data,
                method = "rf",
                tuneGrid = tune_Grid,
                trControl = train_con,
                importance = TRUE,
                ntree=num_tree
)
print(rf_mtry)
#mtry = 4
library(randomForest)
RF_model= randomForest(fare_amount~.,train_data,mtry=4,importance=T)
RF_predict= predict(RF_model,test_data)
RF_model                    
summary(RF_predict)          

fare_amount_pred= cbind(test_data,RF_predict)
test_data= read.csv('test.csv')
test_data= cbind(test_data,RF_predict)
colnames(test_data)[colnames(test_data)=='RF_predict']= 'fare_amount'
write.csv(test_data,'final_output.csv',row.names = F)
