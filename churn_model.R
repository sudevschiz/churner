packages <- c("ggplot2", "dplyr", "ranger", "xgboost", "caret", "e1071","lubridate")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))  
}

# Load the necessary libraries and the dataset
library(ranger)
library(ggplot2)
library(caret)
library(xgboost)
library(lubridate)


########## Required Functions ##############

checkPerformance <- function(pred,y) {
  
  cm <- confusionMatrix(pred,y)
  precision <- posPredValue(pred, y)
  recall <- sensitivity(pred, y)
  F1 <- (2 * precision * recall) / (precision + recall)
  perf.list <- list("confusionMatrix" = cm$table, "F1 score" = F1, "Accuracy" = cm$overall[1])
  return(perf.list)
}

############################################

########### Load the dataset ###############


setwd("~/churner")
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

checkPerformance(rf.pred1$predictions,b_test$status)

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

checkPerformance(rf.pred2$predictions,b_test$status)
############################################

############# Feature Creation #############

#Create features based on the date

d <- master$asOfDate
d_day <- weekdays(d)
d_is_weekend <- d_day %in% c("Saturday","Sunday")
d_month <- months(d)
d_weeknum <- week(d)

#Did the rolling variables increase or decrease?
login_toggle <- master$currLogins > master$prevLogins
ticket_toggle <- master$currTickets > master$prevTickets

#Was the number of broadcasts & campaigns too low? Less than 20 contacts?
broad_amount <- master$BroadCasts20_4weeks - master$broadcast4
camp_amount <- master$Camp20_4weeks - master$Camp2_4weeks

#active contact ratio
active_ratio <- master$ActiveDistinctContacts/(master$Contacts_4weeks+1)

#Bind all these features to the dataframe

master_feature <- cbind(master[,-2],d_day,d_is_weekend,d_month,d_weeknum,login_toggle,ticket_toggle,broad_amount,camp_amount,active_ratio)


rm(d,d_weeknum,d_day,d_is_weekend,d_month,ticket_toggle,login_toggle,broad_amount,camp_amount,active_ratio)


########## Data Sampling (70:15:15) ##########


index <- sample(1:nrow(master_feature),round(0.85*nrow(master_feature)))

#Keep the 15% data as final test data
final_test <- master_feature[-index,]

train <- master_feature[index,]
#Split the training data into positive and negative class
train_p <- train[train$status == 1, ]
train_n <- train[train$status == 0, ]


train_nn <- train_n[sample(1:nrow(train_n),nrow(train_p)),]

m_combined <- rbind(train_p,train_nn)
m_train_index <- sample(1:nrow(m_combined),0.8*nrow(m_combined))
m_train <- m_combined[m_train_index,]
m_val <- m_combined[-m_train_index,]


############ Model Building ########

#Trial 1: random forest on the balanced dataset. 50% positive & 50% negative
rf.trial1 <- ranger(data = m_train[,-1],dependent.variable.name = "status",num.trees = 1000,write.forest = T)
rf.trail1.val <- predict(rf.trial1,data = m_val[,-1])

checkPerformance(rf.trail1.val$predictions,m_val$status)


#Trial 2: xgboost on the balanced dataset. Converting everything to numerics before trial
#Parameters for Classification using xgBoost

label <- as.numeric(as.character(m_train$status))
num_m_train <- sapply(m_train[,-which(colnames(m_train) == "status")], function(x) as.numeric(x))
num_m_val <- sapply(m_val[,-which(colnames(m_train) == "status")], function(x) as.numeric(x))

dtrain <- xgb.DMatrix(data = num_m_train,label = label)
dtest <-  xgb.DMatrix(data = num_m_val)

param_cls <- list(  objective           = "binary:logistic", 
                    # booster = "gblinear",
                    eta                 = 0.1,
                    max_depth           = 10,  # changed from default of 6
                    subsample           = 1,
                    colsample_bytree    = 1,
                    eval_metric         = "error"
                    # alpha = 0.0001, 
                    # lambda = 1
)

#Create the models

xgb_model_cls <- xgb.train(   params              = param_cls, 
                              data                = dtrain,
                              nrounds             = 2000, # changed from 300
                              verbose = 2,
                              
)

xgb_pred <- xgboost::predict(xgb_model_cls, dtest)
xgb_pred <- sapply(xgb_pred , function(x) ifelse(x < 0.5, x <- 0, x <- 1))

checkPerformance(as.factor(xgb_pred),as.factor(m_val$status))
