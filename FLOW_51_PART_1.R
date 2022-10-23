

####### Flow number: 51 | Machine Learning for Business Intelligence 2 (CM) WOAI  ######


###### a) ######

rm(list = ls())
gc()


pre <- read.csv(file = "C:/Users/kaspe/OneDrive - Aarhus universitet/Skrivebord/BI/2. semester/ML2/Exam 2022/dataLC01.csv")
options(scipen = 999) 

# Make loanstatus as factor
loanstatus <- as.factor(pre$loanstatus)
pre$loanstatus <- NULL 
loanstatus_num <- ifelse(loanstatus == "Charged Off", 1, 0) # convert loan status to numerical
pre <- cbind(loanstatus, pre)
str(pre)


###### b) ######

set.seed(202)
options(scipen = 999)

# Split the dataset

train <- sample(nrow(pre), 5000)
pre <- pre[train, ]

pre$homeownershipNONE = NULL
pre$iswv = NULL

###### c) ######

library(caret)
index <- createDataPartition(pre$loanstatus, p = .5,
                             list = FALSE,
                             times = 1)

train <- pre[index, ]
test <- pre[-index, ]

###### d) ######

# load libraries
library(caret)
library(rpart)
library(randomForest)

# define training control
train_control<- trainControl(method = "cv", number = 3)

# train the model 
rf.fit <- train(loanstatus ~ ., data = train, trControl = train_control, method = "rf", ntree = 10)

rf.fit



###### e) ######


train.param <- trainControl(method = "cv", number = 3) # we want to do CV of 3
tune.grid <- expand.grid(n.trees = seq(5,50,500), interaction.depth = 4, shrinkage = 0.1, n.minobsinnode = 10) 


boost.caret.fit <- train(loanstatus ~ ., data = train, 
                         method = "gbm", # boosting method 
                         trControl = train.param,
                         tuneGrid = tune.grid)


###### f) ######



library(xgboost)


xg.train.param <- trainControl(method = "cv", number = 3) 


tune.grid.xgboost <- expand.grid(max_depth = 4:6, gamma = c(0, 1, 2), eta = c(0.03, 0.06), nrounds = 300,
                                 subsample = 0.5, colsample_bytree = 0.1, min_child_weight = 1)


model.xgboost <- train(loanstatus ~ ., train,
                       method = "xgbTree",
                       tuneGrid = tune.grid.xgboost,
                       trControl = xg.train.param,
                       metric = "Accuracy")

model.xgboost # The final values used for the model were nrounds = 300, max_depth = 4, eta = 0.03, gamma = 1, colsample_bytree = 0.1, min_child_weight =
              # 1 and subsample = 0.5.
model.xgboost$results
max(model.xgboost$results$Accuracy)


###### g) ######

# rf

rfmetric <- rbind(max(rf.fit$results$Accuracy), rf.fit$results$Kappa[1])


# boosting

boostmetric <- rbind(boost.caret.fit$results$Accuracy, boost.caret.fit$results$Kappa)

# xgboost

xgboostmetric <- rbind(max(model.xgboost$results$Accuracy), model.xgboost$results$Kappa[2])


list <- as.data.frame(NA)
list <- cbind(rfmetric,boostmetric,xgboostmetric)

list


###### h) ######

loanstatus_num <- ifelse(test$loanstatus == "Charged Off", 1, 0)

# Random forest

rf.pred <- predict(rf.fit, test, type = "prob")
rf.prob <- ifelse(rf.pred$`Fully Paid` < 0.5, 1, 0)

confusionMatrix(as.factor(rf.prob),
                as.factor(loanstatus_num), positive = "1")


# ROC and AUC: a good way to summarize your model
library(caTools) 
colAUC(rf.pred[,1], loanstatus_num, plotROC = TRUE) 




# Boosting


boost.caret.pred <- predict(boost.caret.fit, test, type = "prob") 
boost.caret.prob <- ifelse(boost.caret.pred[,2] < 0.5, 1, 0)


confusionMatrix(as.factor(boost.caret.prob),
                as.factor(loanstatus_num), positive = "1")

# ROC and AUC: a good way to summarize your model
library(caTools) 
colAUC(boost.caret.pred[,2], loanstatus_num, plotROC = TRUE)



# Extreme gradient boosting


xgb.pred <- predict(model.xgboost, test, type = "prob")
xgb.prob <- ifelse(xgb.pred[,2] < 0.5, 1, 0)

confusionMatrix(as.factor(xgb.prob),
                as.factor(loanstatus_num), positive = "1")


# ROC and AUC: a good way to summarize your model
library(caTools) 
colAUC(xgb.pred[,2] , loanstatus_num, plotROC = TRUE) # ROC is the curve, AUC the area under curve



###### i) ######

# Xgboost
var.imp <- varImp(model.xgboost, scale = FALSE)
plot(var.imp)
head(var.imp$importance)



###### j) ######



###### k) ######

prop <- prop.table(table(train$loanstatus))
baseline_acc <- prop[2]
baseline_acc


prop2 <- prop.table(table(test$loanstatus))
baseline_acc <- prop2[2]
baseline_acc

confusionMatrix(as.factor(rf.prob),
                as.factor(loanstatus_num), positive = "1")
confusionMatrix(as.factor(boost.caret.prob),
                as.factor(loanstatus_num), positive = "1")
confusionMatrix(as.factor(xgb.prob),
                as.factor(loanstatus_num), positive = "1")
