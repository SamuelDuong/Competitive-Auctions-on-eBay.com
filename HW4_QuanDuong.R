rm(list = ls())
options(repos = c(CRAN = "http://cran.rstudio.com"))

# install.packages("rpart") # Installation is required for the first of use
library(rpart)
# install.packages("rpart.plot")    # Installation is required for the first of use
library(rpart.plot)
library(rattle)
install.packages("gmodels")
install.packages("hardhat")
install.packages("future")

library(gmodels)
library(caret)
#===================================== classification tree ===========================


ebay.df <- read.csv("C:\\Users\\duong\\OneDrive\\Desktop\\QUAN\\Spring 2024\\Big Data Analytics\\eBayAuctions.csv")
ebay.df <- subset(ebay.df, select=-c(Category, Currency, EndDay)) 
############################################################
### 0. partition the dataset ############
############################################################
set.seed(1)  
train.index <- sample(c(1:dim(ebay.df)[1]), dim(ebay.df)[1]*0.6)  
train.df <- ebay.df[train.index, ]
valid.df <- ebay.df[-train.index, ]

############################################################
### 1. use rpart() to run a classification tree ############
############################################################
competitive.ct <- rpart(Competitive ~ ., data = train.df,
                        control = rpart.control(minsplit = 6),
                    method = "class")
summary(competitive.ct)


newt.ct <- rpart(Competitive ~ OpenPrice + SellerRating + Category.Art.Collectibles + Category.Books + Duration + Currency.nonUS + EndDay.Weekend, data = train.df,
                 control = rpart.control(minsplit = 6),
                 method = "class")

summary(newt.ct)
############################################################
### 2. plot classification tree ############
############################################################
prp(competitive.ct, type = 3, extra = 101,  clip.right.lab = FALSE, box.palette = "GnYlRd", 
    branch = .3, varlen = -10, cex.main=3, space=0)  
# Assuming competitive.ct is your rpart model
fancyRpartPlot(competitive.ct, tweak = 1)


# Print the variable importance
print(modelVarImp)
############################################################
### 3. look at decision rules ############
############################################################
rpart.rules(competitive.ct, extra = 4, cover = TRUE)

############################################################
### 4. evaluate the model ############
competitive.ct.point.pred.train <- predict(competitive.ct,train.df,type = "class")
predicted <- factor(competitive.ct.point.pred.train)
actual <- factor(train.df$Competitive)
# generate confusion matrix for training data
confusionMatrix(as.factor(competitive.ct.point.pred.train), as.factor(train.df$Competitive))
# 4-2. repeat the code for the validation data
competitive.ct.point.pred.valid <- predict(competitive.ct,valid.df,type = "class")
confusionMatrix(as.factor(competitive.ct.point.pred.valid), as.factor(valid.df$Competitive))


#new tree

newt.ct.point.pred.train <- predict(newt.ct,train.df,type = "class")
predicted <- factor(newt.ct.point.pred.train)
actual <- factor(train.df$Competitive)
# generate confusion matrix for training data
confusionMatrix(as.factor(newt.ct.point.pred.train), as.factor(train.df$Competitive))
# 4-2. repeat the code for the validation data
newt.ct.point.pred.valid <- predict(newt.ct,valid.df,type = "class")
confusionMatrix(as.factor(newt.ct.point.pred.valid), as.factor(valid.df$Competitive))

