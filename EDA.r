
setwd("~/Desktop/CFN_ps")
given_data <- read.csv("masterdata_dummy.csv")
str(given_data)

#Setting a random seed for reproducibility
set.seed(82)



given_data$asOfDate <- as.Date(given_data$asOfDate)
#Looking at the first 4 columns
summary(given_data[,1:4])

length(unique(given_data$V1))
#Looks like there are duplicate rows. Unique ID is repeating.
#Storing the duplicated rows for any future purposes
dupes <- given_data[(duplicated(given_data$V1)),]

#Remove the duplicates
given_data <- unique(given_data)
dim(given_data)


#Find the number of unique dates
length(unique(given_data$asOfDate))
#To find the frequency, check the frequency of each date
library(data.table)
uni_dates <- data.table(given_data)[,list(freq = .N),by = asOfDate]
uni_dates

#Hits and PrevHits seems to follow the same pattern. Makes sense as it's a rolling count. (Only one value changes)
#Might be good to create a feature later as to whether the hits increased or decreased in the last cycle
cor(given_data$hits,given_data$prevhits)

#Histogram plots
hist(log10(given_data$hits),breaks = 100, main = "Histogram of log10(hits)")

summary(given_data[,5:9])

hist(given_data$currTickets)
cor(given_data$currTickets,given_data$prevTickets)

#Histogram of tail of the currTicket
hist(given_data$currTickets[given_data$currTickets > 20,],breaks = 10)

summary(given_data[,10:14])

summary(given_data[,15:17])

summary(given_data[,18:23])

given_data[is.na(given_data$emailSent),c(1,2,19:24)]
#Checking the churn ratio in this incomplete data subset
sum(given_data[is.na(given_data$emailSent),c(1,2,19:24)]$status)/nrow(given_data[is.na(given_data$emailSent),c(1,2,19:24)])
#39.7% churn ratio in this small data subset. Shows the significance.

na_data <- given_data[is.na(given_data),]
given_data <- na.omit(given_data)

summary(given_data$status)
#Number of status = 1 gives the total churn cases
sum(given_data$status)
#Check the overall churn ratio
sum(given_data$status)/nrow(given_data)
#8.9% . Shows that this is a highly imbalanced set


#Will analyse churn variable wise

dt <- data.table(given_data)
#Mean of the status variable is the churn ratio
churn_datewise <- dt[,j = mean(status),by = asOfDate]
library(ggplot2)
ggplot(churn_datewise,aes(asOfDate,V1)) + geom_point()


library("corrplot")
corrplot(cor(given_data[,-(1:2)]),type = "upper")
#No Strong correlations. The "Login" variables show ~0.4 correlations.
#More corerlation analysis after features have been created
ggplot(given_data,aes(asOfDate,status)) + geom_smooth()

