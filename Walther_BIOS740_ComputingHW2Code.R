###BIOS 740 Computing HW 2 Code - Andrew Walther

##Packages
#data manipulation/visualizations
library(tidyverse)
#summary statistics
library(mosaic)
#cross-validation
library(caret)
#ML method implementation
library(DynTxRegime)
library(listdtr)
##simulation dataset
#set a seed for reproducibility
set.seed(2019)
#number of samples
n <- 1000
#expit function
expit <- function(x) {1 / (1 + exp(-x))}
#covariates for stage 1
X11 <- rnorm(n)
X12 <- rnorm(n)
#covariates for stage 2
X21 <- rnorm(n)
X22 <- rnorm(n)
#treatment 1 & 2
A1 <- rbinom(n, 1, expit(X11))
A2 <- rbinom(n, 1, expit(X21))
#gamma components of outcome
gamma1 <- A1 * (1 + X11)
gamma2 <- A2 * (1 + X11 + X21 + A1 * X21)
#outcome
Y <- exp(X11) + exp(X21) + X11 * X21 + gamma1 + gamma2 + rnorm(n)
#build dataset and view first few observations
dat = cbind(X11, X12, X21, X22, A1, A2, Y) %>% as.data.frame()
#######################################################################
##Data Processing 
#add column for log transform of Y
dat_log <- dat %>% mutate(log_Y = log(Y))
#Create 10 folds of the dataset and add to TEST/TRAIN lists
folds <- createFolds(dat$Y, 10)
TEST_DATA <- list()
TRAIN_DATA <- list()
for(i in 1:10){
        TEST_DATA[[i]] <- dat[folds[[i]],]
        TRAIN_DATA[[i]] <- dat[-folds[[i]],]}
#######################################################################
##EDA & Feature Selection
#summary stats for Y
favstats(dat$Y)
#summary stats for log(Y)
favstats(dat_log$log_Y)
#hist of Y (right skew is present in the data, some negative values)
dat %>% ggplot() + geom_histogram(aes(x=dat$Y),binwidth = .1) + labs(title = "Histogram of Y outcome",x = "Y outcome")
#hist of log(Y) (now there's left skew in the data, must omit NAs from negative Y's)
dat_log %>% ggplot() + geom_histogram(aes(x=log(Y)),binwidth = 0.1) + labs(title = "Histogram of log Y",x = "Y outcome (log)")
##K-S test for normality (H0: data follows Normal distribution)
#raw data: not normally distributed
ks.test(dat$Y, "pnorm", mean=mean(dat$Y), sd=sd(dat$Y))
#log-transform: still not normally distributed (remove NA/negative values)
dat_log_reduced <- dat_log[complete.cases(dat_log),]
ks.test(dat_log_reduced$log_Y, "pnorm", mean=mean(dat_log_reduced$log_Y), sd=sd(dat_log_reduced$log_Y))
##Feature Selection
#Train RPart Model to compute variable importance
rPartMod <- train(Y~ ., data=dat, method="rpart")
rpartImp <- varImp(rPartMod)
print(rpartImp)
#use all factors since n>>p (X11, X21, A1 seem to be influential)
#######################################################################
##Q-learning
#Initialize lists to hold value/opt trt predictions
Q.estimates <- list()
Q.opt.Tx <- list()
#loop over 10 data folds with Q-learning method
for(i in 1:10){
        #Outcome Model (linear model method for continuous Y, all covariates)
        moMain <- buildModelObj(model = ~X11+X12+X21+X22,
                                solver.method = 'lm')
        moCont <- buildModelObj(model = ~X11+X12+X21+X22,
                                solver.method = 'lm')
        #2nd Stage Analysis (A2 is 2nd stage treatment)
        fitSS <- qLearn(moMain = moMain, moCont = moCont,
                        data = TRAIN_DATA[[i]], 
                        response = TRAIN_DATA[[i]]$Y,  txName = 'A2')
        #outcome model
        moMain <- buildModelObj(model = ~X11+X12+X21+X22,
                                solver.method = 'lm')
        moCont <- buildModelObj(model = ~X11+X12+X21+X22,
                                solver.method = 'lm')
        fitFS <- qLearn(moMain = moMain, moCont = moCont,
                        data = TRAIN_DATA[[i]], 
                        response = fitSS,  txName = 'A1')
        #extract value & optimal treatment predictions for each fold
        Q.estimates[[i]] <- as.numeric(estimator(fitFS))
        Q.opt.Tx[[i]] <- optTx(fitFS, newdata = TEST_DATA[[i]])}
#compute mean & sd for each fold prediction mean
mean.Q <- mean(as.numeric(Q.estimates))
se.Q <- sd(as.numeric(Q.estimates))
#count patients assigned to receive treatment vs. not
trt1.total <- 0
trt0.total <- 0
for(i in 1:10){
        trt1 <- count(Q.opt.Tx[[i]]$optimalTx == 1)
        trt0 <- count(Q.opt.Tx[[i]]$optimalTx == 0)
        trt1.total <- trt1.total + trt1
        trt0.total <- trt0.total + trt0}
#proportion of patients assigned to get treatment
print(trt1.total/(trt1.total+trt0.total))
#######################################################################
##BOWL
#Initialize lists to hold value/opt trt predictions
BOWL.estimates <- list()
BOWL.opt.Tx <- list()
#loop over 10 data folds with BOWL method
for(i in 1:10){
        #2nd Stage Regression - Constant propensity model
        moPropen <- buildModelObj(model = ~ 1,solver.method = 'glm',
                                  solver.args = list('family'='binomial'),
                                  predict.method = 'predict.glm',
                                  predict.args = list(type='response'))
        fitSS <- bowl(moPropen = moPropen,
                      data = TRAIN_DATA[[i]], reward = TRAIN_DATA[[i]]$Y,  txName = 'A2',
                      regime = ~ X11+X12+X21+X22)
        #1st Stage Regression - Constant Propensity model
        fitFS <- bowl(moPropen = moPropen,
                      data = TRAIN_DATA[[i]], reward = TRAIN_DATA[[i]]$Y,  txName = 'A1',
                      regime = ~ X11+X12+X21+X22,
                      BOWLObj = fitSS, lambdas = c(0.5, 1.0), cvFolds = 10)
        #extract value & optimal treatment predictions for each fold
        BOWL.estimates[[i]] <- as.numeric(estimator(fitFS))
        BOWL.opt.Tx[[i]] <- optTx(fitFS, newdata = TEST_DATA[[i]])}
#compute mean & sd for each fold prediction mean
mean.BOWL <- mean(as.numeric(BOWL.estimates))
se.BOWL <- sd(as.numeric(BOWL.estimates))
#count patients assigned to receive treatment vs. not
trt1.total <- 0
trt0.total <- 0
for(i in 1:10){
        trt1 <- count(BOWL.opt.Tx[[i]]$optimalTx == 1)
        trt0 <- count(BOWL.opt.Tx[[i]]$optimalTx == 0)
        trt1.total <- trt1.total + trt1
        trt0.total <- trt0.total + trt0}
#proportion of patients to receive treatment
print(trt1.total/(trt1.total+trt0.total))
#######################################################################
##List-DTR
#feature covariates to base decision rules on
x <- cbind(dat$X11,dat$X12,dat$X21,dat$X22)
stage.x <- rep(1,4)
#Treatments (stage 1 & 2)
a1 <- dat$A1
a2 <- dat$A2
#Outcomes (stage 1 & 2)
y1 <- dat$Y
y2 <- dat$Y
#dynamic treatment regime function (10 folds)
dtr <- listdtr(cbind(y1,y2), cbind(a1,a2), x, stage.x, kfolds = 10, seed = 2019)
#print decision rule If/Then & plot flow chart of decision rules
print(dtr)
plot(dtr)
#predictions for dtr (recommended and optimal)
yrec <- predict(dtr, x, 1)
yopt <- ifelse(x[,1] > 0, 1, 0)
table(yrec, yopt)
#######################################################################