
# Load required packages and libraries

if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
install.packages('rpart.plot')
install.packages('randomForest')


library(rpart)
library(rpart.plot)
library(randomForest)
library(ranger)
library(caret)
library(data.table)
library(caTools)
library(dplyr)
library(tidyverse)
library(pROC)

# Load Credit card transactions data from Github repository


#Set timeout seconds to 150 incase if the download takes more than 60 seconds
getOption('timeout')
options(timeout=150)

# Create temp file and download zip file containing dataset automatically
dw <- tempfile()
download.file("https://github.com/1eeba/Project2-CreditcardFraudDetection/raw/master/creditcard.zip", dw)



# Unzip the files

unzip(dw)

#Read the files 
creditcard_data <- read_csv("creditcard.csv")

#View the loaded data
creditcard_data

#Scale the amount fields
scale(creditcard_data$Amount)

#Splitting the data

library(caTools)
set.seed(123)
data_sample = sample.split(creditcard_data$Class,SplitRatio=0.80)
cc_train_set = subset(creditcard_data,data_sample==TRUE)
cc_test_set = subset(creditcard_data,data_sample==FALSE)



#Scaling the amount field in both the sets


scale(cc_test_set$Amount)
scale(cc_train_set$Amount)


#  Data Visualization
#Dimensions of creditcard dataset

dim(creditcard_data)

#Structure of creditcard dataset
str(creditcard_data)


#First 6 rows

head(creditcard_data)

#Number of Fraudulent and non-fraudulent transactions in the dataset

table(creditcard_data$Class)

# Variance of amount
var(creditcard_data$Amount)

#Standard deviation of amount
sd(creditcard_data$Amount)


#Summary of amount

summary(creditcard_data$Amount)

#Distribution of transactions across time - Transaction versus time

creditcard_data %>%
  ggplot(aes(x = Time, fill = factor(Class))) + 
  geom_histogram(bins = 100) + 
  labs(x = "Time)", y = "Number of transactions", title = "Distribution of transactions across time") +
  facet_grid(Class ~ ., scales = 'free_y') + theme()


#Plot of amount and class
plot(creditcard_data$Amount,creditcard_data$Class)


#Histogram of Class
hist(creditcard_data$Class)


#Histogram of Amount
hist(creditcard_data$Amount)

#Histogram of Amount <1000

hist(creditcard_data$Amount[dataset$Amount < 100])

#Distribution of Amount

ggplot(creditcard_data, aes(x=Amount))+
  scale_x_continuous()+
  geom_density(fill = "blue", alpha = 0.2)+
  theme_minimal()+
  theme(plot.background = element_rect("white"),
        panel.background = element_rect("white"))+
  labs(title = "Distribution of Amount")+
  theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))+
  theme(axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank())+
  xlim(0,1000)

#Histogram of Transaction time and Amount

ggplot(creditcard_data, aes(x=Time))+
  scale_x_continuous()+
  geom_density(fill = "blue", alpha = 0.2)+
  theme_minimal()+
  theme(plot.background = element_rect("white"),
        panel.background = element_rect("white"))+
  labs(title = "Distribution of Time")+
  theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))+
  theme(axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank())


#Plot showing correlation of variables


correlation <- cor(creditcard_data[, -1], method = "pearson")
corrplot(correlation, number.cex = 1, method = "color", type = "full", tl.cex=0.7, tl.col="black")



# Data Modeling

#Model 1 : Logistic Regression model


#Build model on train set
cc_train_set$Class <-factor(cc_train_set$Class)
Model1_Logistic_Model=glm(Class~.,cc_train_set,family=binomial())



#View the summary

summary(Model1_Logistic_Model)

#Plot the result
plot(Model1_Logistic_Model)


#Make predictions
library(pROC)
model1_predictions <- predict(Model1_Logistic_Model, cc_train_set, type='response')
table(cc_train_set$Class, model1_predictions)

#View area under curve
auc.gbm = roc(cc_train_set$Class, model1_predictions, plot = TRUE, col = "blue")


#Applying the model in Test set

#Apply in test set
model1_predictions_test <- predict(Model1_Logistic_Model, newdata = cc_test_set, type = "response")


pred <- ifelse(model1_predictions_test > 0.5, 1, 0)




#View Confusion Matrix
table(cc_test_set$Class, pred)


#Model 2 -Decision tree model

#Build model on train set
decisionTree_model <- rpart(Class ~ . , cc_train_set, method = 'class',minbucket=10)
prp(decisionTree_model)


#Applying on test set
predicted_val <- predict(decisionTree_model, cc_test_set, type = 'class')
probability <- predict(decisionTree_model, cc_test_set, type = 'prob')
prp(decisionTree_model)

#Viewing the Confusion matrix


confusionMatrix(cc_test_set$Class, predicted_val)





#Model 3 -Random forest model


#Build model on train set
set.seed(10)
model3_randomforest <- randomForest(Class ~ ., data = cc_train_set[1:10000,],ntree = 2000, nodesize = 20)

#Applying the model to test set
model3_predict <- predict(model3_randomforest, newdata = cc_test_set)

#Plot the result
varImpPlot(model3_randomforest)
#Viewing the Confusion Matrix
confusionMatrix(cc_test_set$Class, model3_predict)



#Results:
Model1 <- mean(cc_test_set$Class == pred)
Model2 <- mean(cc_test_set$Class == predicted_val)
Model3<- mean(cc_test_set$Class == model3_predict)

Model1
Model2
Model3

