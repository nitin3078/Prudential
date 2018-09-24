library(tidyverse)
library(randomForest)
library(VIM)
library(gbm)
library(mlbench)
library(caret)
library(RCurl)
library(nnet)

# Loading Data from github. This is original dataset

Profile <- read.csv(text=getURL("https://raw.githubusercontent.com/nitin3078/Prudential/master/train.csv"), header=T)

dim(Profile)
glimpse(Profile)



#Summary of response column
summary(Profile$Response)

####summary(profile)
summary(Profile)


#Checking missing values present in dataset

colSums(is.na(Profile))

#Checking if missing values present in "Id" column
sum(is.na(Profile$Id))


# finding column names with NAs in each column
MissingValueColumns<-colnames(Profile)[colSums(is.na(Profile)) > 0]
MissingValueColumns

## Matrix plot. Red for missing values, Darker values are high values.


matrixplot(Profile[MissingValueColumns], interactive = F, sortby = "Employment_Info_1",main="Missing value plot" )


# using purrr to find columns with missing value and arranging columns in ascending order
missingcol <- map_df(Profile, function(x) sum(is.na(x)))
missingcol<- gather(missingcol) 
colnames(missingcol) <-c("variable","missing")

missingcol <- missingcol %>%
  filter( missing>0) %>%
  arrange(missing) %>% mutate( percentmissing = missing/59381*100 ) 
  
print(missingcol)


summary(Profile$Employment_Info_1)




# cleaning Data ( creating new data set to clean , named "ProfileClean"
ProfileClean <- Profile 

dim(ProfileClean)

# Removing 11 out of 13 columns with missing values more than 90%

ProfileClean <- ProfileClean[,-c(39,18,37,30,35,36,38,53,62,70,48)]

# replacing NA with mean for Employment_Info_1

ProfileClean$Employment_Info_1[is.na(ProfileClean$Employment_Info_1)] <- 0.07758

# kNN Impputation for Employment_Info_4
# Not executing below code again as it takes  a lot of time. I have
# already run the code previously and saved the data
# I am uploading data after kNN Imputation once again in ProfileClean data set

###ProfileClean <- knnImputation(ProfileClean, k = 3, scale = T, meth = "weighAvg", distData = NULL)

# Loading Data from github. This is Cleaned dataset, after median Imputaion and kNN Imputation

ProfileClean <- read.csv(text=getURL("https://raw.githubusercontent.com/nitin3078/Prudential/master/ProfileClean.csv"), header=T)

sum(is.na(ProfileClean))

#Updating first column name to "Id"
colnames(ProfileClean)[1] <- "Id"


### Exploratory data analysis


# Tidying data and creating custom variables

# Correlation between BMI, Wt and Ht
cor(ProfileClean$BMI,ProfileClean$Wt)
cor(ProfileClean$BMI,ProfileClean$Ht) 
cor(ProfileClean$Ht,ProfileClean$Wt) 

# BMI is highly correlated with Wt, removing Wt from dataset
ProfileClean <- subset(ProfileClean, select = -c(Wt) )

# as columns  Medical_Keyword_1 to Medical_Keyword_48 are 0 and 1 with little overall information
# , creating a new column as a sum of all these column : MedKeywordSum

ProfileClean$MedKeywordSum <- rowSums(ProfileClean[,c(68:115)])
# dropping Medical_Keyword_1 to Medical_Keyword_48 from dataset
ProfileClean <- subset(ProfileClean, select = -c(68:115) )

# as columns  Medical_History_3 to Medical_History_41 are 0 and 1 with little overall information
# , creating a new column as a sum of all these column : MedHistSum

ProfileClean$MedHistSum <- rowSums(ProfileClean[,c(33:67)])
# dropping Medical_Keyword_1 to Medical_Keyword_48 from dataset
ProfileClean <- subset(ProfileClean, select = -c(33:67) )
 
# Distribution of custom column MedKeywordSum is not normal
ggplot(ProfileClean, aes(x=MedKeywordSum)) + 
  geom_histogram(aes(y = ..density..),position="identity", colour="black", alpha=0.2, bins = 10)+ 
  ggtitle("Histogram of Custom Column MedHistSum") +
  theme_bw() + stat_function(fun = dnorm, colour = "red",
                             args = list(mean = mean(ProfileClean$MedKeywordSum, na.rm = TRUE),
                                         sd = sd(ProfileClean$MedKeywordSum, na.rm = TRUE)))


# Distribution of custom column MedHistSum is normal
ggplot(ProfileClean, aes(x=MedHistSum)) + 
  geom_histogram(aes(y = ..density..),position="identity", colour="black", alpha=0.2, bins = 10)+
  ggtitle("Histogram of Custom Column MedHistSum") +
  theme_bw() + stat_function(fun = dnorm, colour = "red",
                             args = list(mean = mean(ProfileClean$MedHistSum, na.rm = TRUE),
                                        sd = sd(ProfileClean$MedHistSum, na.rm = TRUE)))



# Boxplot Response ~ BMI
ggplot(ProfileClean, aes(x=factor(Response), y=BMI)) + ggtitle("Boxplot Response ~ BMI") + 
  geom_boxplot(colour="blue")+  theme(axis.text.x=element_text(angle=90,hjust=1))

# We can see that BMI has prediction power. Response value 8 has lower value of BMI compared to response value 1.



# Boxplot Response ~ Ht
ggplot(ProfileClean, aes(x=factor(Response), y=Ht)) + ggtitle("Boxplot Response ~ Ht") + 
  geom_boxplot(colour="red")+  theme(axis.text.x=element_text(angle=90,hjust=1))
#We can see that Ht has very less prediction power. 




# Boxplot Response ~ Ins_Age
ggplot(ProfileClean, aes(x=factor(Response), y=Ins_Age)) + ggtitle("Boxplot Response ~ Ins_Age") + 
  geom_boxplot(colour="blue")+  theme(axis.text.x=element_text(angle=90,hjust=1))
# From this plot, we can see Ins_Age has some predictive power

# Boxplot Response ~ Product_Info_4
ggplot(ProfileClean, aes(x=factor(Response), y=Product_Info_4)) + ggtitle("Boxplot Response with custom variable MedHistSum") + 
  geom_boxplot(colour="blue")+  theme(axis.text.x=element_text(angle=90,hjust=1))
# From this plot, we can see Product_Info_4 has much predictive power


# Boxplot Response ~ MedKeywordSum
ggplot(ProfileClean, aes(x=factor(Response), y=MedKeywordSum)) + ggtitle("Boxplot Response with custom column MedKeywordSum") + 
  geom_boxplot(colour="red")+  theme(axis.text.x=element_text(angle=90,hjust=1))

# Little predictive power

# Boxplot Response ~ MedHistSum
ggplot(ProfileClean, aes(x=factor(Response), y=MedHistSum)) + ggtitle("Boxplot Response with custom variable MedHistSum") + 
  geom_boxplot(colour="blue")+  theme(axis.text.x=element_text(angle=90,hjust=1))

# little predictive power MedHistSum



# histogram of Response on Product_Info_2

ggplot(ProfileClean, aes(x=Response)) +  ggtitle("Histogram  Response with Product_Info_2 categories") + 
  geom_histogram(position="identity", colour="blue", alpha=0.2, bins = 10)+
  facet_grid(. ~ Product_Info_2)
  theme(axis.text.x=element_text(angle=90,hjust=1))
 
  
 # From above histogram, we can see that distribution of response variable is dependent on Product_Info_2
  
  
  # histogram of Response on Family_Hist_1
  
  ggplot(ProfileClean, aes(x=Response)) + ggtitle("Histogram  Response with Family_Hist_1 values") +
    geom_histogram(position="identity", colour="black", alpha=0.2, bins = 10)+
    facet_grid(. ~ Family_Hist_1)
  theme(axis.text.x=element_text(angle=90,hjust=1))
  
  # From above histogram, we can see that distribution of response variable is dependent on Family_Hist_1
  
  

# loading "test" data from website and then removing columns just like we did in Train data set to make
# algorithms work

test <- read.csv(text=getURL("https://raw.githubusercontent.com/nitin3078/Prudential/master/test.csv"), header=T)

# execute below steps to delete required columns just like we did in train data set to make it congruent

test <- test[,-c(39,18,37,30,35,36,38,53,62,70,48)]
test <- subset(test, select = -c(Wt) )
test$MedKeywordSum <- rowSums(test[,c(68:115)])
test <- subset(test, select = -c(68:115) )
test$MedHistSum <- rowSums(test[,c(33:67)])
test <- subset(test, select = -c(33:67) )



# Multinomial model creation and submission

MultinomModel <- multinom(Response ~  Product_Info_4+ Product_Info_2 +  BMI + Ins_Age +Family_Hist_1+ MedKeywordSum + MedHistSum, data = ProfileClean)
MultinomModel



predict_Response <- predict (MultinomModel, test, "probs")

test$Response <- predict (MultinomModel, test)


# Create a file to submit on Kaggle, which contain only 2 columns, Id and Response
submission <- test[, c(1,35)]
  
write.csv(submission, "C:\\Users\\nitin\\Desktop\\study\\Special topics in BANA R\\Project\\submission.csv", row.names = F)
  
# accuracy is 0.36562 
  