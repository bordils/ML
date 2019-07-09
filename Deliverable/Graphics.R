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

library(ggplot2)
library(plotly)
library(gridExtra)


##############################################################
#################### Functions        ########################
##############################################################


trainObj_to_F1 <- function(model) {
  conf <- caret::confusionMatrix(model, norm="none")
  
  tp <- conf$table[2,2]
  tn <- conf$table[2,2]
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

##################################################################################################################################

##################################################
######### Categorical Value Exploration ##########
##################################################






# plotting HISTOGRAMS WITHOUT YES NO SEP
##################################################
# age
agehist <- ggplot(refinedData, aes(x=age)) + 
  geom_histogram() +
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# avg_balance
avg_balanceHist <- ggplot(refinedData, aes(x=avg_balance)) + 
  geom_histogram() +
  theme_minimal() +
  theme(legend.position="none",
        axis.title.y = element_blank(),
        axis.title.x = element_text(size=8),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# duration_lc
duration_lcHist <- ggplot(refinedData, aes(x=duration_lc)) + 
  geom_histogram() +
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# amt_contact_campaign
ACCHist <- ggplot(refinedData, aes(x=amt_contact_campaign)) + 
  geom_histogram() +
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# amt_preCampaign
APCHist <- ggplot(refinedData, aes(x=amt_preCampaign)) + 
  geom_histogram() +
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )
gridHistPre<-grid.arrange(agehist, avg_balanceHist, duration_lcHist, 
                          ACCHist, APCHist, ncol = 5, heights=c(1, 3))


##################################################################################################################################



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


# plotting HISTOGRAMS WITHOUT YES NO SEP
##################################################
# age
agehist2 <- ggplot(refinedData, aes(x=age)) + 
  geom_histogram() +
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# avg_balance
avg_balanceHist2 <- ggplot(refinedData, aes(x=avg_balance)) + 
  geom_histogram() +
  theme_minimal() +
  theme(legend.position="none",
        axis.title.y = element_blank(),
        axis.title.x = element_text(size=8),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# duration_lc
duration_lcHist2 <- ggplot(refinedData, aes(x=duration_lc)) + 
  geom_histogram() +
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# amt_contact_campaign
ACCHist2 <- ggplot(refinedData, aes(x=amt_contact_campaign)) + 
  geom_histogram() +
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# amt_preCampaign
APCHist2 <- ggplot(refinedData, aes(x=amt_preCampaign)) + 
  geom_histogram() +
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )
gridHistPre2<-grid.arrange(agehist2, avg_balanceHist2, duration_lcHist2, 
                          ACCHist2, APCHist2, ncol = 5, heights=c(1, 3))


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


## Cluster Size Decision for K= 4 Clusters
pam_fit <- pam(gower_dist, diss = TRUE, k = 4)


x <- scaled_Dist$points[,1]
y <- scaled_Dist$points[,2]
plot(x, y, xlab="Coordinate 1", ylab="Coordinate 2",
     main="Multidimensional Scaled data", type="n")
text(x, y, labels = row.names(clusterSample[,nonTarget]), col=pam_fit$clustering, cex=.7)

clusterFrame = as.data.frame(cbind(x, y, pam_fit$clustering))

#### put this in a data frame and paint it

clusterFrame$V3<-as.factor(clusterFrame$V3)

clusterGraph<-ggplot(clusterFrame, aes(x=clusterFrame$x, y=clusterFrame$y, color=clusterFrame$V3 )) + 
  geom_point(size=2 )+
  #stat_density_2d(aes(fill = clusterFrame$V3), geom="polygon") +
  theme_minimal() +
  #stat_ellipse() +
  ggtitle("PAM Clustering of Multidimensional Scaling") +
  labs(colour = "Clusters") +
  theme(legend.position=c(0.05,0.8),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )
  clusterGraph

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


####
x <- scaled_Dist$points[,1]
y <- scaled_Dist$points[,2]
plot(x, y, xlab="Coordinate 1", ylab="Coordinate 2",
     main="Multidimensional Scaled data", type="n")
text(x, y, labels = row.names(clusterSample[,nonTarget]), col=pam_fit$clustering, cex=.7)



clusterplot<-scaled(refinedData, aes(x=amt_contact_campaign, y=amt_preCampaign)) + 
  geom_point(size=2, aes(color=subscription_target))+
  #coord_cartesian(xlim =c(-5, 13), ylim = c(-1, 8))+
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_text(size = 8),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )



##################################################
##################################################
##################################################
##################################################
##################################################

#scatter plot and correlation of numerical features.
#frequencies for categorical features.

##################################################
######### Categorical Value Exploration ##########
##################################################

#good one in case we cant fit htis
aes(job, fill = subscription_target)

##### job 1
jobplot<-ggplot(refinedData, aes(job, fill = subscription_target)) +
  geom_histogram(stat="count") + #position dodge to put side by side
  coord_flip() +
  theme_minimal() +
  theme(legend.position="none",
        #axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.y = element_text(angle = 0, size = 6)
        )

##### marital_status 2
maritalplot<-ggplot(refinedData, aes(marital_status, fill = subscription_target)) +
  geom_histogram(stat="count", position = "dodge") + #position dodge to put side by side
  coord_flip() +
  theme_minimal() +
  theme(legend.position="none",
        #axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.y = element_text(angle = 60, size = 8)
  )

##### education 3
educationplot<-ggplot(refinedData, aes(education, fill = subscription_target)) +
  geom_histogram(stat="count") + #position dodge to put side by side
  coord_flip() +
  theme_minimal() +
  theme(legend.position="none",
        #axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.y = element_text(angle = 60, size = 8)
  )

##### credit_default 4
creditplot<-ggplot(refinedData, aes(credit_default, fill = subscription_target)) +
  geom_histogram(stat="count") + #position dodge to put side by side
  coord_flip() +
  theme_minimal() +
  theme(legend.position="none",
        #axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.y = element_text(angle = 60, size = 8)
  )

##### housing_loan 5 
housingplot<-ggplot(refinedData, aes(housing_loan, fill = subscription_target)) +
  geom_histogram(stat="count") + #position dodge to put side by side
  coord_flip() +
  theme_minimal() +
  theme(legend.position="none",
        #axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.y = element_text(angle = 60, size = 8)
  )

##### personal_loan 6 
ploanplot<-ggplot(refinedData, aes(personal_loan, fill = subscription_target)) +
  geom_histogram(stat="count") + #position dodge to put side by side
  coord_flip() +
  theme_minimal() +
  theme(legend.position="none",
        #axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.y = element_text(angle = 60, size = 8)
  )

##### contact_type 7 
contactplot<-ggplot(refinedData, aes(contact_type, fill = subscription_target)) +
  geom_histogram(stat="count") + #position dodge to put side by side
  coord_flip() +
  theme_minimal() +
  theme(legend.position="none",
        #axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.y = element_text(angle = 60, size = 8)
  )

##### day_lc 8
daylcplot<-ggplot(refinedData, aes(day_lc, fill = subscription_target)) +
  geom_histogram(stat="count") + #position dodge to put side by side
  coord_flip() +
  theme_minimal() +
  theme(legend.position="none",
        #axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.y = element_text(angle = 0, size = 6)
  )


##### month_lc 9 
monthlcplot<-ggplot(refinedData, aes(month_lc, fill = subscription_target)) +
  geom_histogram(stat="count") + #position dodge to put side by side
  coord_flip() +
  theme_minimal() +
  theme(legend.position="none",
        #axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.y = element_text(angle = 0, size = 6)
  )

##### outcome_last_camp 10 
lastcplot<-ggplot(refinedData, aes(outcome_lastCampaign, fill = subscription_target)) +
  geom_histogram(stat="count") + #position dodge to put side by side
  coord_flip() +
  theme_minimal() +
  theme(legend.position="none",
        #axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.y = element_text(angle = 60, size = 8)
  )

##### cat_days_lc 11
catdayplot<-ggplot(refinedData, aes(cat_days_lc, fill = subscription_target)) +
  geom_histogram(stat="count") + #position dodge to put side by side
  coord_flip() +
  theme_minimal() +
  theme(legend.position="none",
        #axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.y = element_text(angle = 0, size = 6)
  )

##### create blank plot to fith them all
  ## here we must draw the legend

blankPlot <- ggplot()+geom_blank(aes(1,1))+
  theme(plot.background = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        panel.border = element_blank(),
        panel.background = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_blank(), 
        axis.text.y = element_blank(),
        axis.ticks = element_blank()
  )

grid.arrange(jobplot, maritalplot, educationplot, housingplot, ploanplot, 
             contactplot, daylcplot, creditplot,  monthlcplot, lastcplot, catdayplot, blankPlot, ncol=3)

#ncol=2, nrow=2, widths=c(1, 1), heights=c(1, 1)

# gs <- grobTree(jobplot, maritalplot, educationplot,
#        creditplot, housingplot, ploanplot,
#        contactplot, daylcplot, monthlcplot, lastcplot, catdayplot)

gs <- lapply(1:9, function(ii) 
  grobTree(rectGrob(gp=gpar(fill=ii, alpha=0.5)), textGrob(ii)))
grid.arrange(grobs=gs, ncol=4, 
             top="top label", bottom="bottom\nlabel", 
             left="left label", right="right label")
grid.rect(gp=gpar(fill=NA))

##################################################
######### Categorical Value Exploration ##########
##################################################

# plotting density for numerical values
ageageDensity <- ggplot(refinedData, aes(x=age, fill=subscription_target)) + 
  geom_density(alpha=.5) +
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
    )


# avg_balance
avg_balanceDensity <- ggplot(refinedData, aes(x=avg_balance, fill=subscription_target)) + 
  geom_density(alpha=.5) +
  coord_cartesian(xlim =c(-5, 5), ylim = c(0, 1.6))+
  theme_minimal() +
  theme(legend.position="none",
        axis.title.y = element_blank(),
        axis.title.x = element_text(size=8),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# duration_lc
duration_lcDensity <- ggplot(refinedData, aes(x=duration_lc, fill=subscription_target)) + 
  geom_density(alpha=.5) +
  coord_cartesian(xlim =c(-5, 5), ylim = c(0, 0.5))+
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# amt_contact_campaign
amt_contact_campaignDensity <- ggplot(refinedData, aes(x=amt_contact_campaign, fill=subscription_target)) + 
  geom_density(alpha=.5) +
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# amt_preCampaign
amt_preCampaignDensity <- ggplot(refinedData, aes(x=amt_preCampaign, fill=subscription_target)) + 
  geom_density(alpha=.5) +
  coord_cartesian(xlim =c(0, 5), ylim = c(0, 1))+
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )
amt_preCampaignDensity

gridDensity<-grid.arrange(ageDensity, avg_balanceDensity, duration_lcDensity, 
                          amt_contact_campaignDensity, amt_preCampaignDensity, ncol = 5, heights=c(1, 3))


gridDensity


# plotting paired numerical values
# age VS Av balance
ageAVbal<-ggplot(refinedData, aes(x=age, y=avg_balance)) + 
  geom_point(size=2, aes(color=subscription_target))+
  coord_cartesian(xlim =c(-3, 3), ylim = c(-5, 12))+
  theme_minimal() +
  theme(legend.position="none",
      axis.title.x = element_text(size = 8),
      axis.title.y = element_text(size = 8),
      axis.text.x = element_text(angle = 0, size = 6),
      axis.text.y = element_text(angle = 0, size = 6)
  )

# age VS duration last call
ageDuration<-ggplot(refinedData, aes(x=age, y=duration_lc)) + 
  geom_point(size=2, aes(color=subscription_target))+
  #coord_cartesian(xlim =c(-3, 3), ylim = c(-5, 12))+
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_text(size = 8),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# av Balance VS duration LC
avBalDuration<-ggplot(refinedData, aes(x=avg_balance, y=duration_lc)) + 
  geom_point(size=2, aes(color=subscription_target))+
  coord_cartesian(xlim =c(-5, 13), ylim = c(-6, 4))+
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_text(size = 8),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# age VS amount CC
ageACC<-ggplot(refinedData, aes(x=age, y=amt_contact_campaign)) + 
  geom_point(size=2, aes(color=subscription_target))+
  #coord_cartesian(xlim =c(-5, 13), ylim = c(-6, 4))+
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_text(size = 8),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# av Balance VS amount CC
avBalACC<-ggplot(refinedData, aes(x=avg_balance, y=amt_contact_campaign)) + 
  geom_point(size=2, aes(color=subscription_target))+
  coord_cartesian(xlim =c(-5, 13), ylim = c(-1, 5))+
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_text(size = 8),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# duration VS amount CC
durationACC<-ggplot(refinedData, aes(x=duration_lc, y=amt_contact_campaign)) + 
  geom_point(size=2, aes(color=subscription_target))+
  #coord_cartesian(xlim =c(-5, 13), ylim = c(-1, 5))+
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_text(size = 8),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# age VS amount PC
ageAPC<-ggplot(refinedData, aes(x=age, y=amt_preCampaign)) + 
  geom_point(size=2, aes(color=subscription_target))+
  #coord_cartesian(xlim =c(-5, 13), ylim = c(-1, 5))+
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_text(size = 8),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# av Balance VS amount PC
avBalAPC<-ggplot(refinedData, aes(x=avg_balance, y=amt_preCampaign)) + 
  geom_point(size=2, aes(color=subscription_target))+
  coord_cartesian(xlim =c(-5, 13), ylim = c(-1, 8))+
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_text(size = 8),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

# duration LC VS amount PC
durationAPC<-ggplot(refinedData, aes(x=duration_lc, y=amt_preCampaign)) + 
  geom_point(size=2, aes(color=subscription_target))+
  #coord_cartesian(xlim =c(-5, 13), ylim = c(-1, 8))+
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_text(size = 8),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )

#amount CC VS amount PC
ACCAPC<-ggplot(refinedData, aes(x=amt_contact_campaign, y=amt_preCampaign)) + 
  geom_point(size=2, aes(color=subscription_target))+
  #coord_cartesian(xlim =c(-5, 13), ylim = c(-1, 8))+
  theme_minimal() +
  theme(legend.position="none",
        axis.title.x = element_text(size = 8),
        axis.title.y = element_text(size = 8),
        axis.text.x = element_text(angle = 0, size = 6),
        axis.text.y = element_text(angle = 0, size = 6)
  )


#first row
agetext<-grobTree( rectGrob(gp=gpar(fill="dark grey")), textGrob("Age", gp=gpar()))

corCoef12<-cor(refinedData$age, refinedData$avg_balance)
corCoef13<-cor(refinedData$age, refinedData$duration_lc)
corCoef14<-cor(refinedData$age, refinedData$amt_contact_campaign)
corCoef15<-cor(refinedData$age, refinedData$amt_preCampaign)

AgeAvBalCor<-grobTree( rectGrob(gp=gpar(fill="light grey")), textGrob("0.106", gp=gpar()))
ageDurationCor<-grobTree( rectGrob(gp=gpar(fill="light grey")), textGrob("-0.013", gp=gpar()))
ageACCCor<-grobTree( rectGrob(gp=gpar(fill="light grey")), textGrob("0.029", gp=gpar()))
ACCAPCCor<-grobTree( rectGrob(gp=gpar(fill="light grey")), textGrob("-0.001", gp=gpar()))


#second row
#ageAVbal
avBaltext<-grobTree( rectGrob(gp=gpar(fill="dark grey")), textGrob("Av. Balance", gp=gpar()))

corCoef23<-cor(refinedData$avg_balance, refinedData$duration_lc)
corCoef24<-cor(refinedData$avg_balance, refinedData$amt_contact_campaign)
corCoef25<-cor(refinedData$avg_balance, refinedData$amt_preCampaign)

avBalDurationCor<-grobTree( rectGrob(gp=gpar(fill="light grey")), textGrob("0.028", gp=gpar()))
avBalACCCor<-grobTree( rectGrob(gp=gpar(fill="light grey")), textGrob("-0.023", gp=gpar()))
avBalAPCCor<-grobTree( rectGrob(gp=gpar(fill="light grey")), textGrob("0.043", gp=gpar()))

#third row
#ageDuration
#avBalDuration
durationText<-grobTree( rectGrob(gp=gpar(fill="dark grey")), textGrob("Call Duration", gp=gpar()))

corCoef34<-cor(refinedData$duration_lc, refinedData$amt_contact_campaign)
corCoef35<-cor(refinedData$duration_lc, refinedData$amt_preCampaign)

durationACCCor<-grobTree( rectGrob(gp=gpar(fill="light grey")), textGrob("-0.168", gp=gpar()))
durationAPCCor<-grobTree( rectGrob(gp=gpar(fill="light grey")), textGrob("0.014", gp=gpar()))

#fourth row
#ageACC
#avBalACC
#durationACC
ACCtext<-grobTree( rectGrob(gp=gpar(fill="dark grey")), textGrob("ACC", gp=gpar()))

corCoef45<-cor(refinedData$duration_lc, refinedData$amt_contact_campaign)
ACCAPCCor<-grobTree( rectGrob(gp=gpar(fill="light grey")), textGrob("-0.168", gp=gpar()))

#fifth row
#ageAPC
#avBalAPC
#durationAPC
#ACCAPC
APCtext<-grobTree( rectGrob(gp=gpar(fill="dark grey")), textGrob("APC", gp=gpar()))

# title <- grobTree( rectGrob(gp=gpar(fill="dark grey")), textGrob("APC", gp=gpar()))
# grid.draw(title)

grid.arrange(ageAVbal,ageDuration,avBalDuration,
             ageACC, avBalACC,durationACC,
             ageAPC,avBalAPC, durationAPC, ncol=3)

gridCor<-grid.arrange(agetext,AgeAvBalCor,ageDurationCor,ageACCCor, ACCAPCCor,
                      ageAVbal, avBaltext, avBalDurationCor, avBalACCCor, avBalAPCCor,
                      ageDuration, avBalDuration, durationText, durationACCCor, durationAPCCor,
                      ageACC, avBalACC, durationACC, ACCtext, ACCAPCCor,
                      ageAPC, avBalAPC, durationAPC, ACCAPC, APCtext,
                      ncol=5
                      )
#############################################################################
#############################################################################
#############################################################################

dummyplot<-ggplot(refinedData, aes(job, fill = subscription_target)) +
  geom_histogram(stat="count") + #position dodge to put side by side
  coord_flip() +
  theme_minimal() +
  theme(legend.position=c(0.8,0.8),
        #axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.y = element_text(angle = 0, size = 6)
  )

legend <- cowplot::get_legend(dummyplot)

blankPlot<-grid.draw(legend)
show(blankPlot)

#############################################################################
#############################################################################
#############################################################################
##################################################
##################################################
##################################################

