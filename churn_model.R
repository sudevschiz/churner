# Load the necessary libraries and the dataset
library(ranger)
library(ggplot2)
library(caret)
library(xgboost)


########## Required Functions ##############

checkPerformance <- function(pred,y) {
  
  cm <- confusionMatrix(pred$predictions,y)
  precision <- posPredValue(pred$predictions, y)
  recall <- sensitivity(pred$predictions, y)
  F1 <- (2 * precision * recall) / (precision + recall)
  perf.list <- list("confusionMatrix" = cm$table, "F1 score" = F1, "Accuracy" = cm$overall[1])
  return(perf.list)
}

############################################

########### Load the dataset ###############


setwd("~/Desktop/CFN_ps")
master <- read.csv("masterdata_dummy.csv")
str(master)

############################################

########### Data preparation ###############
#Remove duplicates
master <- unique(master)
#Convert the dates
master$asOfDate <- as.Date(master$asOfDate)
#status is a factor.
master$status <- as.factor(master$status)

############################################

########### Data imputation ################

#As a first step, just remove the data with NA values
#master <- na.omit(master)
#To include the information from the NA valued observation, apply the median value to NA
#To provide more information, imputed values for positive and negative class to be diff
med_p <- lapply(na.omit(master[master$status == 1,18:23]), median)
med_n <- lapply(na.omit(master[master$status == 0,18:23]), median)

master[is.na(master$ActiveDistinctContacts) & master$status == 1,18] <- med_p[1]
master[is.na(master$emailSent) & master$status == 1,19] <- med_p[2]
master[is.na(master$Tasks) & master$status == 1,20] <- med_p[3]
master[is.na(master$Tags) & master$status == 1,21] <- med_p[4]
master[is.na(master$Opportunities) & master$status == 1,22] <- med_p[5]
master[is.na(master$WebForms) & master$status == 1,23] <- med_p[6]

master[is.na(master$ActiveDistinctContacts) & master$status == 0,18] <- med_p[1]
master[is.na(master$emailSent) & master$status == 0,19] <- med_n[2]
master[is.na(master$Tasks) & master$status == 0,20] <- med_n[3]
master[is.na(master$Tags) & master$status == 0,21] <- med_n[4]
master[is.na(master$Opportunities) & master$status == 0,22] <- med_n[5]
master[is.na(master$WebForms) & master$status == 0,23] <- med_n[6]

summary(master[18:23])

############################################

########### Benchmarking [very primitive models on the entire dataset] #####

#Benchmark - 1
#Split the data into two sets 80:20::train:test

index <- sample(1:nrow(master),round(0.8*nrow(master)))
b_train <- master[index,]
b_test <- master[-index,]

#Run a sample random forest alogrithm on the entire dataset. 
rf.bench1 <- ranger(data = b_train[,3:24],dependent.variable.name = "status",num.trees = 300,write.forest = T)
rf.pred1 <- predict(rf.bench1,data = b_test[,3:24])

checkPerformance(rf.pred1,b_test$status)

#Benchmark - 2
#Split the data set to create a balanced data set and run the tests
master_status_1 <- master[master$status == 1, ]
master_status_0 <- master[master$status == 0, ]
dim(master_status_1)
b_train_n <- master_status_0[sample(1:dim(master_status_0)[1],size = dim(master_status_1)[1]),]
b_train_p <- master_status_1
bal_master <- rbind(b_train_n,b_train_p)

index <- sample(1:nrow(bal_master),round(0.8*nrow(bal_master)))
b_train <- bal_master[index,]
b_test <- bal_master[-index,]

rm(master_status_0,master_status_1,b_train_p,b_train_n,bal_master)
#Run a sample random forest alogrithm on the balanced dataset. 50% positive & 50% negative
rf.bench2 <- ranger(data = b_train[,3:24],dependent.variable.name = "status",num.trees = 300,write.forest = T)
rf.pred2 <- predict(rf.bench2,data = b_test[,3:24])

checkPerformance(rf.pred2,b_test$status)
############################################




