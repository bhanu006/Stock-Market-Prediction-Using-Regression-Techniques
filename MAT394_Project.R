# Stock Market Prediction Using Regression Algorithms

#importing libraries
library(ggplot2)
library(e1071)
library(caret)
library(cowplot)
library(gridExtra)
library(glmnet)
library(mlbench)
library(psych)
library(pROC)

# Data loading and preprossesing
fb2 <- read.csv(file = "FB.csv")
fb2$Date <- as.Date(fb2$Date)

#Exploratory Data Analysis
summary(fb2)
str(fb2)
names(fb2)
head(fb2,5)
tail(fb2,5)
sum(is.na(fb2)) #prints the number of null values
cor(fb2[,2:7]) #Correlation matrix

ggplot(data = fb2, mapping = aes(Date, Adj.Close)) + geom_point() +
  geom_line() + xlab("Date") + ylab("Adj.Close") + ggtitle("Facebook Stock Prices") 




#Split the data into train and test - use data before year 2019 as train data, and 2019 onwards as test data
fb2_split <-  split(fb2, fb2$Date < as.Date("2019-02-02"))
fb2_split$`FALSE`

training <- as.data.frame(fb2_split$`TRUE`)
test <-  as.data.frame(fb2_split$`FALSE`)

#LinearRegression
#Train model
lm <- lm(Adj.Close ~ Date , data = training)
summary(lm)
coefficients(lm)
plot(fb2$Date,fb2$Adj.Close)
abline(lm, col="red")

#Predict on test data
lm_pred <- predict(lm, test)
#R Squared Value
R2_lm = R2(lm_pred,test$Adj.Close)
R2_lm

#Comparing Actual vs Predicted Values
A = cbind(test$Adj.Close[1:5],lm_pred[1:5])
newheaders <- c("Actual Values","Predicted Values")
colnames(A) <- newheaders
A

#Plotting predicted Vs Actual Values
plot_lm = ggplot() + geom_line(data = test, aes(x = Date, y = test$Adj.Close), color = "blue") +
  geom_line(data = test, aes(x = Date, y = lm_pred), color = "red") +
  xlab('Dates') +
  ylab('Adj.Close')
print(plot_lm)

#RidgeRegression
# Custom Control Parameters
custom <- trainControl(method = "repeatedcv",
                       number = 10 ,
                       repeats = 5,
                       verboseIter = T)
#Train model
set.seed(1234)
ridge <- train(Adj.Close ~ .,training, method = "glmnet",tuneGrid = expand.grid(alpha = 0,lambda = seq(0.0001,1,length = 100)), trControl = custom)

#Predict on test data
rr_pred <- predict(ridge, newdata = test)

#Comparing Actual vs Predicted Values
B = cbind(test$Adj.Close[1:5],rr_pred[1:5])
newheaders <- c("Actual Values","Predicted Values")
colnames(B) <- newheaders
B
#R Squared Value
R2_rr = R2(rr_pred,test$Adj.Close)
R2_rr

#Plotting predicted Vs Actual Values
plot_rr = ggplot() + geom_line(data = test, aes(x = Date, y = test$Adj.Close), color = "blue") +
  geom_line(data = test, aes(x = Date, y = rr_pred), color = "red") +
  xlab('Dates') +
  ylab('Adj.Close')+ggtitle("Actual vs Predicted for Ridged Regression") 

print(plot_rr)


# KNN Model
trControl <- trainControl(method = 'repeatedcv',
                          number = 10,
                          repeats = 3)
#Train model
set.seed(333)
knn <- train(Adj.Close ~.,
             data = training,
             tuneGrid = expand.grid(k=1:70),
             method = 'knn',
             metric = 'Rsquared',
             trControl = trControl,
             preProc = c('center', 'scale'))

# Model Performance
knn
plot(knn)
#Predict on test data
knn_pred <- predict(knn, newdata = test)
plot(knn_pred ~ test$Adj.Close)
#R Squared Value
R2_knn = R2(knn_pred,test$Adj.Close)
R2_knn

#Comparing Actual vs Predicted Values
C = cbind(test$Adj.Close[1:5],knn_pred[1:5])
newheaders <- c("Actual Values","Predicted Values")
colnames(C) <- newheaders
C

#Plotting predicted Vs Actual Values
plot_knn = ggplot() + 
  geom_line(data = test, aes(x = Date, y = test$Adj.Close), color = "blue") +
  geom_line(data = test, aes(x = Date, y = knn_pred), color = "red") +
  xlab('Dates') +
  ylab('Adj.Close')

print(plot_knn)

#Randomforest
#Splitting data
library(randomForest) #can be used to perform RF as well as Bagging
set.seed(1)
rf_train=c(1:989) 
rf_test=fb2[-rf_train ,"Adj.Close"]
#Train model
#Using mtry=3
set.seed(1)
rf=randomForest(Adj.Close~.,data=fb2,subset=rf_train,mtry=3,importance =TRUE)

#Predict on test data
rf_pred = predict(rf ,newdata=fb2[-rf_train ,])
mean((rf_pred-rf_test)^2)
#R Squared Value
R2_rf = R2(rf_pred,rf_test)
R2_rf

#Comparing Actual vs Predicted Values
D = cbind(test$Adj.Close[1:5],rf_pred[1:5])
newheaders <- c("Actual Values","Predicted Values")
colnames(D) <- newheaders
D

#Plotting predicted Vs Actual Values
plot_rf = ggplot() + 
  geom_line(data = fb2[-rf_train,], aes(x = Date, y = rf_test), color = "blue") +
  geom_line(data = fb2[-rf_train,], aes(x = Date, y = rf_pred), color = "red") +
  xlab('Dates') +
  ylab('Adj.Close')

print(plot_rf)
