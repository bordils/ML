####################################################################################
####################################################################################
#################### Machine Learning Project Exploratory scripts ##################
####################    Mickey the Machine and Homeless Hans      ##################
####################################################################################
####################################################################################

###Needed libraries
library(plyr)
library(FactoMineR)
library(factoextra)
library(dummies)
library(caTools)
library(chemometrics)
library(MASS)
library(glmnet)
library(DMwR)
library(caret)
library(klaR)
library(class)
library(randomForest)
library(dplyr)
library(cluster)
library(RSNNS)
library(neural)

library(e1071)
library(kernlab)


##############################################################
#################### Functions        ########################
##############################################################


trainObj_to_F1 <- function(model) {
  conf <- caret::confusionMatrix(model, norm="none")
  
  tp <- conf$table[2,2]
  tn <- conf$table[1,1]
  fp <- conf$table[2,1]
  fn <- conf$table[1,2]
  
  recall <- tp / (tp +fn) #recall= TP/ (TP+FN)
  precision <- tp/ (tp + fp)#precision =TP /(TP+FP ) 
  return( 2 * (precision * recall) / (precision + recall))
}



##############################################################
#################### Data Loading     ########################
##############################################################
rawData<-read.csv("Bank_Marketing_data.csv", header=TRUE, sep=",")
colnames(rawData) <- c("age", "job", "marital_status", "education", "credit_default", "avg_balance", 
                       "housing_loan", "personal_loan", "contact_type", "day_lc", "month_lc", 
                       "duration_lc", "amt_contact_campaign", "days_since_lc", "amt_preCampaign", "outcome_lastCampaign", 
                       "subscription_target")

summary(rawData)


#######################################################
#########     Missings          #######################
#######################################################

##Check for missing values:
missings <- c()
missings <- sum(apply(rawData, 1, is.na))
infinites <- sum(apply(rawData, 1, is.infinite))
#Not a single NA in the data


#########################################################
########### Multivariate Outlier Detection    ###########
#########################################################



#Multivariate Outlier detection
outliers <- Moutlier(rawData[,c(1, 6, 12, 13)], quantile = 0.975, plot = TRUE)
md <- as.data.frame(outliers$md)
colnames(md)[1] <- "value"
rd <- as.data.frame(outliers$rd)
colnames(rd)[1] <- "value"



#These are the outliers
par(mfrow=c(1,1))
mdOutliers <- subset(md, value >= outliers$cutoff)
rdOutliers <- subset(rd, value >= outliers$cutoff)

mvOutliers <- unique(rbind(mdOutliers, rdOutliers))
mvOutliers <- rawData[row.names(mvOutliers) ,c(1, 6, 12, 13)]
mvOutliers <- as.data.frame(na.omit(mvOutliers))
outliersV2 <- Moutlier(mvOutliers, quantile = 0.975, plot = TRUE)

mdV2 <- as.data.frame(outliersV2$md)
colnames(mdV2)[1] <- "value"
rdV2 <- as.data.frame(outliersV2$rd)
colnames(rdV2)[1] <- "value"

mdOutliers <- subset(mdV2, value >= outliersV2$cutoff)
rdOutliers <- subset(rdV2, value >= outliersV2$cutoff)
nrow(mdOutliers)
nrow(rdOutliers)




rm(outliers,  rd, md,  infinites, missings, rdV2, mdV2, mvOutliers)

#Remove outliers or not
# refinedData <- rawData[as.numeric(row.names(mdOutliers)) * -1,]
# refinedData <- rawData[as.numeric(row.names(rdOutliers)) * -1,] 

#It is decided in accordance with Prof. Belanche to not exclude outliers
refinedData <- rawData




#########################################################
########### FACTORIZE ALL Categorical Columns ###########
#########################################################



refinedData$job <- as.factor(refinedData$job)
refinedData$marital_status <- as.factor(refinedData$marital_status)
refinedData$education <- as.factor(refinedData$education)
refinedData$credit_default <- as.factor(refinedData$credit_default)
refinedData$housing_loan <- as.factor(refinedData$housing_loan)
refinedData$personal_loan <- as.factor(refinedData$personal_loan)
refinedData$contact_type <- as.factor(refinedData$contact_type)
refinedData$outcome_lastCampaign <- as.factor(refinedData$outcome_lastCampaign)
refinedData$subscription_target <- as.factor(refinedData$subscription_target)
refinedData$month_lc <- as.factor(refinedData$month_lc)#months are categories as well (e.g. we don't know the year)
refinedData$day_lc <- as.factor(refinedData$day_lc)# even though the days look numeric they are actually categorical
levels(refinedData$subscription_target) <- c( "no","yes")

##Categorize last day of contact
refinedData$cat_days_lc <- 999
for(i in 1:nrow(refinedData)){
  if(refinedData[i,"days_since_lc"] <= -1){
    refinedData[i,"cat_days_lc"] <- 0
  }else if(refinedData[i,"days_since_lc"] <= 30){
    refinedData[i,"cat_days_lc"] <- 1
  }else if(refinedData[i,"days_since_lc"] <= 90){
    refinedData[i,"cat_days_lc"] <- 2
  }else if(refinedData[i,"days_since_lc"] <= 180){
    refinedData[i,"cat_days_lc"] <- 3
  }else if(refinedData[i,"days_since_lc"] <= 365){
    refinedData[i,"cat_days_lc"] <- 4
  }else {
    refinedData[i,"cat_days_lc"] <- 5
  }
}
refinedData$cat_days_lc <- as.factor(refinedData$cat_days_lc)
levels(refinedData$cat_days_lc) <- c("never", "month", "quartal", "half_year", "year", "before")
count(refinedData$cat_days_lc)
refinedData <- refinedData[,!(colnames(refinedData) == 'days_since_lc')]





##############################################################
#################### Log + Standardize  ######################
##############################################################
numericCols = which(sapply(refinedData, is.numeric))


for(i in numericCols){
  if(min(refinedData[,i]) <= 0){
    refinedData[,i] <- log(refinedData[,i] - min(refinedData[,i]) + 1)
  }else{#e.g. age --> no need for that
    refinedData[,i] <- scale(log(refinedData[,i]))
  }
  hist(refinedData[,i], main=paste("hist of", colnames(refinedData)[i]))
}

refinedData[,numericCols] <- scale(refinedData[,numericCols])
summary(refinedData[,numericCols])



##############################################################
#################### Data Exploration ########################
##############################################################
# MCA with excluded continuous values
targetCol = which( colnames(refinedData)=="subscription_target" )
nonTarget = which( !colnames(refinedData)=="subscription_target" )
numericCols = which(sapply(refinedData, is.numeric))
catCols = which(sapply(refinedData, is.factor))
catCols = catCols[catCols != 16]

par(mfrow=c(1,1))
mcaObj <- MCA(refinedData[,catCols], graph =TRUE)#TODO recheck columns..

#Dimension 1 --> subscription target nearly on same heigth day_lc, ducation, job, housing loan
# dim 2 --> credit default, personal loan, martial status and contact type are in the same area

for(i in numericCols){
  hist(refinedData[,i],  main = paste("Histogram of" , colnames(refinedData)[i]))
}#--> avg_balance, duratin_lc (quite ok after log), amt_contact_campaign, amt_preCampaign are not really gaussian


# rm(mcaObj)
source ("acm.r")
mc <- acm(refinedData[,catCols])
plot(mc$rs[,1],mc$rs[,2],col=refinedData$subscription_target)
barplot(mc$vaps)
i <- 1
while (mc$vaps[i] > mean(mc$vaps)) i <- i+1
(nd <- i-1)
refinedPsi <- as.data.frame(cbind(mc$rs[,1:nd], refinedData$subscription_target))
colnames(refinedPsi)[length(colnames(refinedPsi))] = "subscription_target"
refinedPsi$subscription_target <- as.factor(refinedPsi$subscription_target)
levels(refinedPsi$subscription_target) <- c( "no","yes")


##############################################################
####################  PAM Clustering      ####################
##############################################################
# 
set.seed(12345)
clusterSample <-  sample_n(refinedData, 5000)# with 45k and internal dummy conversion --> run out of RAM
plyr::count(clusterSample$subscription_target)

gower_dist <- daisy(clusterSample[,nonTarget],
                    metric = "gower")

scaled_Dist <- cmdscale(gower_dist,eig=TRUE, k=2)

x <- scaled_Dist$points[,1]
y <- scaled_Dist$points[,2]
plot(x, y, xlab="Coordinate 1", ylab="Coordinate 2",
     main="Multidimensional Scaled data", type="n")
text(x, y, labels = row.names(clusterSample[,nonTarget]), cex=.7)


#Using shilloute as Clustering perfomance metric: https://www.r-bloggers.com/clustering-mixed-data-types-in-r/
sil_width <- c(NA)

for(i in 2:10){

  pam_fit <- pam(gower_dist, diss = TRUE, k = i)
  pam_fit <- pam(gower_dist, diss = TRUE, k = i)

  sil_width[i] <- pam_fit$silinfo$avg.width

}

# Plot performance

plot(1:10, sil_width, xlab = "Number of clusters", ylab = "Silhouette Width")
lines(1:10, sil_width)


## Cluster Size Decision for K= 4 Clusters
pam_fit <- pam(gower_dist, diss = TRUE, k = 4)

pam_results <- clusterSample %>%
  mutate(cluster = pam_fit$clustering) %>%
  group_by(cluster) %>%
  do(the_summary = summary(.))


pam_results$the_summary

x <- scaled_Dist$points[,1]
y <- scaled_Dist$points[,2]
plot(x, y, xlab="Coordinate 1", ylab="Coordinate 2",
     main="Multidimensional Scaled data", type="n")
text(x, y, labels = row.names(clusterSample[,nonTarget]), col=pam_fit$clustering, cex=.7)

#Results: The results show that the clustering does not produce clusters with a particularly high yes/no ratio different than
#the original one. The origninal ratio is about 1:8 and the clusters show:
# 1- 1:5
# 2- 1:20
# 3- 1:6
# 4- 1:5
# The only knowledge we infer is that observation of cluster 2 are more unlikely to take a credit this cluster is defined by:
# Age is lower than the average
# The clusters profession is dominated by blue-collar workers. Small amount of management, retired, technicians etc.
# Education --> very little tertiary educated people
# many housing loans
# previous campaign had been normally unkown. This makes us think that they had probably been excluded in the last campaigns for a reason 
#since they are not very likely to take a credit or maybe the credit will not be approved. This would need further analysis by
#the domain experts.

rm(gower_dist)

#####################################################################
### Feature selection based on Random FOrest variable importance  ###
#####################################################################

#Since most feature selection methods are not applicable on this problem
#And not relevant inormation has been eliminated as well we  CONSIDERED relying on the results of the importance of variables
#in the random forest model BUT we discarded this idea again after talking to Prof. Belanche
#Due to the bad performance or martial_status and credit_default we considered excluding those variables, but we came to the
#conclusion we are not sure about it so we added them again

#Furthermore we need to discard duration_lc since we will not have this information at the moment we do the prediction.

dropMe <- c( "duration_lc") 
refinedData <- refinedData[,!(names(refinedData) %in% dropMe)]
rm(dropMe)
#Update Metadata
targetCol = which( colnames(refinedData)=="subscription_target" )
nonTarget = which( !colnames(refinedData)=="subscription_target" )
numericCols = which(sapply(refinedData, is.numeric))

##############################################################
#################### Data Splitting   ########################
##############################################################
set.seed(12345)
train_ind <- sample(1:nrow(refinedData), 2/3* nrow(refinedData))
test_ind <- setdiff(1:nrow(refinedData), train_ind)

train <- refinedData[train_ind,]
test <- refinedData[test_ind,]


##############################################################
#################### Data Balancing   ########################
##############################################################

#This means the data is highly unbalanced --> Nearly 40k not taking the loan and 5k taking the loan
#--> oversampling needed
set.seed(12345)

balData <-  SMOTE(subscription_target ~ ., train, perc.over = 700,perc.under=115) #TODO USE the big dataset for final training

balDataSmall <- sample_n(train[train$subscription_target=="yes",], 3500, replace = FALSE)
balDataSmall <- rbind(balDataSmall, sample_n(train[train$subscription_target=="no",], 3500, replace = FALSE))
balDataSmall <- sample_n(balDataSmall, 7000, replace = FALSE )#Data needs to be shuffled... #TODO CHANGE TO 10000

plyr::count(balData$subscription_target)
plyr::count(balDataSmall$subscription_target)

#TODO replace refined Data with balData
train <- balData

trainSmall <- balDataSmall

rm(balData, balDataSmall)

############################################
###### Create Dummy variables    ##########
############################################
#DUMMY creation needed due to RBF network package RSNNS
dummyData<-dummy.data.frame(trainSmall,names=c("job", "marital_status","education","credit_default","housing_loan",
                                               "personal_loan","contact_type", "cat_days_lc", "outcome_lastCampaign",
                                               "day_lc", "month_lc"))


plyr::count(dummyData$subscription_target)







#################################################################################################################################
#################################################################################################################################
####################################  Modelling Part     ########################################################################
#################################################################################################################################
#################################################################################################################################









train.results <- matrix (rep(0,2*12),nrow=12)
colnames (train.results) <- c("Models", "F1_Score")
train.results[,"Models"] <- c("GLM", "LDA", "QDA", "RDA", "KNN", "NB", "ANN.MLP", "ANN.RBF", "SVM.LIN", "SVM.POLY", "SVM.RBF", "RF")
train.results[,2] <- 0
train.results <- as.data.frame(train.results)
train.results$F1_Score <- as.numeric(0)

#update metadata
targetCol = which( colnames(train)=="subscription_target" )
nonTarget = which( !colnames(train)=="subscription_target" )
numericCols = which(sapply(train, is.numeric))

targetColDummy = which( colnames(dummyData)=="subscription_target" )
nonTargetDummy = which( !colnames(dummyData)=="subscription_target" )



##############################################################
################## GLM Logistic regression  ##################
##############################################################
set.seed(12345)



ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

glm.model <- caret::train(subscription_target~.,  data=train, method="glm", family="binomial",
                       trControl = ctrl)
train.results[train.results$Models=="GLM",2] <- trainObj_to_F1(glm.model)

# (conf <- confusionMatrix(glm.cv, norm="none"))
# 
# 
#  tp <- conf$table[2,2]
#  tn <- conf$table[2,2]
#  fp <- conf$table[2,1]
#  fn <- conf$table[1,2]
#  
#  recall <- tp / (tp +fn) #recall= TP/ (TP+FN)
#  precision <- tp/ (tp + fp)#precision =TP /(TP+FP ) 
#  
#  train.results[train.results$Models=="GLM",2] <- 2 * (precision * recall) / (precision + recall)
#  
#  
#  rm(glm.cv, conf, tp, tn, fp, fn, recall, precision)

##############################################################
##################  LDA, QDA, RDA, KNN     ###################
##############################################################
set.seed(12345)
#KNN
# method = 'kknn' --> with #k, distance and kernel
#method = 'knn' --> with #k
ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

knn.model <- caret::train(subscription_target~.,  data=train, method="knn", trControl = ctrl)
train.results[train.results$Models=="KNN",2] <- trainObj_to_F1(knn.model)




#LDA

ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

lda.model <- caret::train(subscription_target~.,  data=train, method="lda", trControl = ctrl)
train.results[train.results$Models=="LDA",2] <- trainObj_to_F1(lda.model)

# (conf <- confusionMatrix(lda.model, norm="none"))
# 
# tp <- conf$table[2,2]
# tn <- conf$table[2,2]
# fp <- conf$table[2,1]
# fn <- conf$table[1,2]
# 
# recall <- tp / (tp +fn) #recall= TP/ (TP+FN)
# precision <- tp/ (tp + fp)#precision =TP /(TP+FP ) 
# train.results[train.results$Models=="LDA",2] <- 2 * (precision * recall) / (precision + recall)
# rm(conf, tp, tn, fp, fn, recall, precision)

#QDA

qda.model <- caret::train(subscription_target~.,  data=train, method="qda", trControl = ctrl)
train.results[train.results$Models=="QDA",2] <- trainObj_to_F1(qda.model)

#RDA

rda.model <- caret::train(subscription_target~.,  data=train, method="rda", trControl = ctrl)
train.results[train.results$Models=="RDA",2] <- trainObj_to_F1(rda.model)





##############################################################
##################  Naive Bayes ##############################
##############################################################
set.seed(12345)
ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

nb.model <- caret::train(subscription_target~.,  data=train, method="nb", trControl = ctrl)
train.results[train.results$Models=="NB",2] <- trainObj_to_F1(nb.model) 



##############################################################
##################  ANN  - MLP & RBF   #######################
##############################################################
set.seed(12345)
ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

#MLP
mlp.model <- caret::train(subscription_target~.,  data=train, method="mlp", trControl = ctrl)
train.results[train.results$Models=="ANN.MLP",2] <- trainObj_to_F1(mlp.model) 


#RBF
# rbf.model <- caret::train(subscription_target~. ,  data=dummyData, method='rbf', trControl = ctrl)
# train.results[train.results$Models=="ANN.RBF",2] <- trainObj_to_F1(rbf.model)


# inputs <- as.matrix(seq(0,10,0.1))
# outputs <- as.matrix(sin(inputs) + runif(inputs*0.2))
# outputs <- normalizeData(outputs, "0_1")
# 
# model <- RSNNS::rbf(inputs, outputs, size=40, maxit=1000, 
#              initFuncParams=c(0, 1, 0, 0.01, 0.01), 
#              learnFuncParams=c(1e-8, 0, 1e-8, 0.1, 0.8), linOut=TRUE)




trainCV<-dummyData[sample(nrow(dummyData)),] 
folds <- cut(seq(1,nrow(trainCV)),breaks=5,labels=FALSE)

(numberNeurons <- c( 15, 20, 30))
rbf.results <- matrix (rep(0,2*length(numberNeurons)),nrow=length(numberNeurons))
colnames (rbf.results) <- c("nodes", "F1Score")
rbf.results[,"nodes"] <- numberNeurons
rbf.results[,"F1Score"] <- 0
rbf.results <- as.data.frame(rbf.results)

#Perform 5 fold cross validation
for(amtNeuron in numberNeurons){
  for(i in 1:5){
    #Segement your data by fold using the which() function
    indexes <- which(folds==i,arr.ind=TRUE)
    testData <- trainCV[indexes, ]
    trainData <- trainCV[-indexes, ]
    
    target <-as.data.frame(as.numeric(trainData[,"subscription_target"]))
    
    
    # data <- rbftrain (trainData[,nonTargetDummy], amtNeuron, target, visual = FALSE)#TODO how to evaluate #neurons
    
    model.rsnns.rbf <- RSNNS::rbf(trainData[,nonTargetDummy], target, size=amtNeuron, maxit=1000, 
                                  initFuncParams=c(0, 1, 0, 0.01, 0.01), 
                                  learnFuncParams=c(1e-8, 0, 1e-8, 0.1, 0.8), linOut=TRUE)
    # model.rsnss.rbfdda <- rbfDDA(trainData[,nonTargetDummy], target)
    
    
    #preds <- rbf (testData[,nonTargetDummy], data$weight, data$dist, data$neurons, data$sigma)
    rbf.preds <- predict(model.rsnns.rbf, testData[,nonTargetDummy])
    # rbfdda.preds <-predict(model.rsnss.rbfdda, testData[,nonTargetDummy])# WTF Are those results
    
    rbf.preds <- round(rbf.preds)
    conf <- RSNNS::confusionMatrix(testData$subscription_target, rbf.preds)
    cv.error <- 1- ((conf[1,1] + conf[2,2]) / sum(conf))
    
    tp <- conf[2,2]
    tn <- conf[1,1]
    fp <- conf[1,2]
    fn <- conf[2,1]
    
    recall <- tp / (tp +fn) #recall= TP/ (TP+FN)
    precision <- tp/ (tp + fp)#precision =TP /(TP+FP ) 
    
    rbf.results[rbf.results$nodes==amtNeuron, "F1Score"] <- rbf.results[rbf.results$nodes==amtNeuron, "F1Score"] + ( 2 * (precision * recall) / (precision + recall))
    
    rm(rbf.preds, conf, cv.error, recall, precision)
    
  }
  rbf.results[rbf.results$nodes==amtNeuron, "F1Score"] <- rbf.results[rbf.results$nodes==amtNeuron, "F1Score"] / 5
  
}

train.results[train.results$Models=="ANN.RBF",2] <- rbf.results[which.max(rbf.results$F1Score), "F1Score"]


##############################################################
##################  SVMs  Crossvalidated      ################
##############################################################
set.seed(12345)
ctrl <- trainControl(method = "repeatedcv", number = 5, savePredictions = TRUE)
#LINEAR KERNEL
# method = 'svmLinear'
#TODO really needed?

svm.lin.model <- caret::train(subscription_target~.,  data=trainSmall, method="svmLinear3", trControl = ctrl)
train.results[train.results$Models=="SVM.LIN",2] <- trainObj_to_F1(svm.lin.model) 

#POLYNOMIAL KERNEL
# method = 'svmPoly'
svm.poly.model <- caret::train(subscription_target~.,  data=trainSmall, method="svmPoly", trControl = ctrl)
train.results[train.results$Models=="SVM.POLY",2] <- trainObj_to_F1(svm.poly.model) 


#RBF KERNEL
# method = 'svmRadial' # trains sigma and Cost
# method = 'svmRadialSigma' # same but more nattorw sigma search
svm.rbf.model <- caret::train(subscription_target~.,  data=trainSmall, method="svmRadial", trControl = ctrl)
train.results[train.results$Models=="SVM.RBF",2] <- trainObj_to_F1(svm.rbf.model) 




##############################################################
##################  Random Forest     ########################
##############################################################
set.seed(12345)
(ntrees <- round(10^seq(1,3.6,by=0.2)))

rf.results <- matrix (rep(0,2*length(ntrees)),nrow=length(ntrees))
colnames (rf.results) <- c("ntrees", "OOB")
rf.results[,"ntrees"] <- ntrees
rf.results[,"OOB"] <- 0

ii <- 1

for (nt in ntrees)
{ 
  print(nt)
  
  model.rf <- randomForest(subscription_target ~ ., data = train, ntree=nt, proximity=FALSE, 
                           sampsize=c(yes=1000, no=1000), strata=train$subscription_target)
  
  # get the OOB
  rf.results[ii,"OOB"] <- model.rf$err.rate[nt,1]
  
  ii <- ii+1
}


rf.results
lowest.OOB.error <- as.integer(which.min(rf.results[,"OOB"]))
(ntrees.best <- rf.results[lowest.OOB.error,"ntrees"])

model.rf <- randomForest(subscription_target ~ ., data = train, ntree=ntrees.best, 
                         sampsize=c(yes=1000, no=1000), strata=train$subscription_target,
                         proximity=FALSE, importance=TRUE)


model.rf$confusion#CHecked that the lines are true and the columnes are predictions!
recall <- model.rf$confusion[2,2] / (model.rf$confusion[2,2] + model.rf$confusion[2,1])#recall= TP/P
precision <- model.rf$confusion[2,2] / (model.rf$confusion[2,2] + model.rf$confusion[1,2])#precision =TP /(TP+FP ) 

train.results[train.results$Models=="RF",2] <- 2 * (precision * recall) / (precision + recall)


# The importance of variables
importance(model.rf)
varImpPlot(model.rf)

plot(model.rf)






##############################################################
##################  FINAL TeSTING  ###########################
##############################################################

#The best model is:
train.results[which.max(train.results$F1_Score),]
#RANDOM FOREST! 

#In this case we don't need to retrain the model with full data since RF is always created with the full dataset 
#We only need to check the generalization error therefore

#RF
pred.rf.final <- predict(model.rf, test[,nonTarget], type="class")

(final.conf <- caret::confusionMatrix(pred.rf.final, test$subscription_target, positive="yes"))


tp <- final.conf$table[2,2]
tn <- final.conf$table[1,1]
fp <- final.conf$table[2,1]
fn <- final.conf$table[1,2]

recall <- tp / (tp +fn) #recall= TP/ (TP+FN)
precision <- tp/ (tp + fp)#precision =TP /(TP+FP ) 
(final.F1 <-( 2 * (precision * recall) / (precision + recall)))


final.confusionMatrix <- matrix(c("Actual Value", "No", "yes", "Pred No",  tn, fn, "Pred Yes", fp, tp),ncol=3, nrow=3)
final.performanceIndicators <-t(final.conf$overall)
final.byClassPerformance<-t(final.conf$byClass)

rm(tp, tn, fp, fn, recall, precision)

#The performance of Randomforests is terrible. It seems to terribly overfit the data

