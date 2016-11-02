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


########## General Functions ##############

checkPerformance <- function(pred,y) {
  
  cm <- confusionMatrix(pred,y)
  precision <- posPredValue(pred, y)
  recall <- sensitivity(pred, y)
  F1 <- (2 * precision * recall) / (precision + recall)
  perf.list <- list("confusionMatrix" = cm$table, "F1 score" = F1, "Accuracy" = cm$overall[1])
  return(perf.list)
}


########### Load the dataset ###############


setwd("~/churner")
master <- read.csv("masterdata_dummy.csv")
str(master)


########### Data preparation ###############
#Remove duplicates
master <- unique(master)
#Convert the dates
master$asOfDate <- as.Date(master$asOfDate)
#status is a factor.
master$status <- as.factor(master$status)


########### Data imputation ################

#As a first step, just remove the data with NA values
master <- na.omit(master)
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


########### Outlier Treatment ############

m <- master

m[m$hits > 100,]$hits <- 100
m[m$prevhits > 100,]$prevhits <- 100
m[m$spamComplaints > 500,]$spamComplaints <- 500
m[m$emailSent > 1000,]$emailSent <- 1000

master <- m

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


########## Initial Data Sampling  ##########


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

#Model 1: random forest on the balanced dataset. 50% positive & 50% negative

model_1_rf <- function(train, val, n_trees = 300) {
  rf.trial1 <- ranger(data = train[,-1],dependent.variable.name = "status",num.trees = n_trees,importance = "impurity",write.forest = T)
  rf.trail1.val <- predict(rf.trial1,data = val[,-1])
  
  return(c(checkPerformance(rf.trail1.val$predictions,val$status),rf.trial1))
  
}

rfm <- model_1_rf(m_train,m_val,n_trees = 1000)



#Model 2: xgboost on the balanced dataset. Converting everything to numerics before trial

model_2_xgb <- function(train,val,depth = 50,eta = 0.1,nrounds = 300) {
  num_m_train <- sapply(m_train[,-which(colnames(m_train) %in% c("status","V1"))], function(x) as.numeric(x))
  num_m_val <- sapply(m_val[,-which(colnames(m_train)  %in% c("status","V1"))], function(x) as.numeric(x))
  
  dtrain <- xgb.DMatrix(data = num_m_train,label = as.numeric(as.character(m_train$status)))
  dtest <-  xgb.DMatrix(data = num_m_val, label = as.numeric(as.character(m_val$status)))
  
  #Parameters for Classification using xgBoost
  param_cls <- list(  objective           = "binary:logistic", 
                      # booster = "gblinear",
                      eta                 = eta,
                      max_depth           = depth,  # changed from default of 6
                      subsample           = 1,
                      colsample_bytree    = 1,
                      eval_metric         = "error"
                      # alpha = 0.0001, 
                      # lambda = 1
  )
  
  #Create the models
  
  xgb_model_cls <- xgb.train(   params              = param_cls, 
                                data                = dtrain,
                                nrounds             = nrounds, # changed from 300
                                verbose = 2,
                                nthreads            = 8,
                                watchlist = list(validation = dtest)
                                
  )
  
  xgb_pred <- xgboost::predict(xgb_model_cls, dtest)
  xgb_pred <- sapply(xgb_pred , function(x) ifelse(x < 0.5, x <- 0, x <- 1))
  
  #Checking importance
  
  #xgb_imp <- xgb.importance(model = xgb_model_cls,feature_names = attributes(num_m_train)$dimnames[[2]])
  
  return(c(checkPerformance(as.factor(xgb_pred),as.factor(m_val$status)),xgb_model_cls))
}


model_2_xgb(m_train,m_val,depth = 170, nr = 1000, eta = 0.02)





######## Customised Undersampling ########
#Earlier sampling was a selection of 50% random samples form the positive and negative class.
#Lot of information from the negative class lost in this case.
#So, create several datasets with balanced distribution, apply the models on them,ensemble the models



#nrow_p number of samples have to selected multiple times without repetation

index_n <- sample(1:nrow(train_n),size = 0.15*nrow(train_p))
index_p <- sample(1:nrow(train_p),size = 0.15*nrow(train_p))

test_n <- train_n[index_n,]
test_p <- train_p[index_p,]

train_n <- train_n[-index_n,]
train_p <- train_p[-index_p,]

test_np <- rbind(test_n,test_p)

list_of_trainsets <- vector("list",n_sets )
nrow_p <- nrow(train_p)
nrow_n <- nrow(train_n)

n_sets <- ceiling(nrow_n/nrow_p)
samplesize <- ceiling(nrow_n/n_sets)

nrow_subset_1 <- 1
for(i in 1:n_sets){
  if(i*samplesize > nrow_n) {
    nrow_subset_2 <- nrow_n
  }
  else {
    nrow_subset_2 <- i*samplesize
  }
  
  list_of_trainsets[[i]] <- rbind(train_n[1:nrow_subset_2,],train_p)
  nrow_subset_1 <- nrow_subset_2+1
}

## Model execution on subsets ##

list_of_models <- vector("list",n_sets )
acc <- 0
for(i in 1:(n_sets)){
  
  list_of_models[[i]] <- model_1_rf(list_of_trainsets[[1]],test_np,n_trees = 1000)
  acc <- acc+ list_of_models[[i]]$Accuracy
}

mean_acc <- acc/n_sets


######## Ensembling models ####### #TBD#

pred_list <- vector("list",n_sets)

for (i in 1:n_sets){
  pred_list[[i]] <- predict(list_of_models[[i]]$forest,data = final_test)$predictions
}

checkPerformance(pred_list[[1]],final_test$status)

######## Final Test ###########

#
num_final_test <- sapply(final_test[,-which(colnames(final_test) == "status")], function(x) as.numeric(x))
fdtest <- xgb.DMatrix(data = num_final_test, label = as.numeric(as.character(final_test$status)))
xgb_pred <- xgboost::predict(xgb_model_cls, fdtest )
xgb_pred <- sapply(xgb_pred , function(x) ifelse(x < 0.5, x <- 0, x <- 1))

checkPerformance(as.factor(xgb_pred),as.factor(final_test$status))


