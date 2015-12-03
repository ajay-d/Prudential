library(readr)
library(dplyr)
library(tidyr)
library(rstan)
library(Matrix)
library(xgboost)
library(ggplot2)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10) 

train.full <- read_csv("data/train.csv.zip")
test.full <- read_csv("data/test.csv.zip")
sample_submission <- read_csv("data/sample_submission.csv.zip")

names(train.full)

train.full %>%
  select(1:12) %>%
  complete.cases %>%
  sum

train.full %>%
  select(matches("Insurance_History")) %>%
  names

#complete
train.full %>%
  select(matches("Product_Info")) %>%
  complete.cases %>%
  sum

train.full %>%
  select(matches("InsuredInfo")) %>%
  complete.cases %>%
  sum

#NOT complete
train.full %>%
  select(matches("Employment_Info")) %>%
  complete.cases %>%
  sum
train.full %>%
  select(matches("Insurance_History")) %>%
  complete.cases %>%
  sum

ggplot(train.full) + geom_histogram(aes(Response))
ggplot(train.full) + geom_histogram(aes(Medical_History_2))
ggplot(train.full) + geom_histogram(aes(Medical_Keyword_1))

table(train.full$Response)
table(train.full$Product_Info_5)
table(train.full$Product_Info_7)

table(train.full$Family_Hist_1)
table(train.full$Product_Info_4)

summary(train.full$Ret_MinusTwo)
summary(train.full$Ret_MinusOne)
summary(train.full$Weight_Daily)

train.sample <- train.full %>%
  select(2:12, -Product_Info_2, matches("InsuredInfo")) %>%
  sample_n(1000) %>%
  as.matrix

trainMatrix <- train.full %>%
  select(2:12, -Product_Info_2, matches("InsuredInfo")) %>%
  as.matrix

y <- train.full %>%
  select(Response) %>%
  mutate(Response=Response-1) %>%
  as.matrix

#"rank:pairwise" 
#"multi:softmax"
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 8)

parallel::detectCores()

cv.nround <- 5
cv.nfold <- 3

bst.cv <- xgb.cv(param=param, data = trainMatrix, label = y, 
                 nfold = cv.nfold, nrounds = cv.nround)

nround <- 50
##Better form for training data
dtrain <- xgb.DMatrix(data = trainMatrix, label = y)

bst <- xgboost(param = param, data = trainMatrix, label = y, nrounds = nround, nthread = 8)
pred <- predict(bst, as.matrix(trainMatrix))

# verbose = 1, print evaluation metric
bst <- xgboost(data = dtrain, nrounds = nround, verbose = 1)

# verbose = 2, also print information about tree
bst <- xgboost(data = dtrain, nrounds = nround, verbose = 2)

#linear boosting
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 8, "nthread" = 8)
bst <- xgb.train(param = param, data = dtrain, nrounds = nround, verbose = 2)
bst <- xgb.train(param = param, data = dtrain, nrounds = nround, verbose = 2, booster = "gblinear")


# Get the feature real names
names <- dimnames(trainMatrix)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = bst)

# Nice graph
xgb.plot.importance(importance_matrix[1:10,])


##Create dummy vars
train.sample <- train.full %>%
  select(2:12, -Product_Info_2, matches("InsuredInfo")) %>%
  sample_n(1000) %>%
  as.matrix

train.full.sparse <- train.full %>%
  select(Family_Hist_1, Response) %>%
  mutate(Response=Response-1) %>%
  sample_n(200)


table(train.full.sparse$Family_Hist_1)
levels(train.full.sparse$Family_Hist_1)
train.full.sparse <- as.data.frame(lapply(train.full.sparse, as.factor))

t <- sparse.model.matrix(Response~.-1, train.full.sparse)
head(t)

##Create all dummy vars
cat.vars <- bind_rows(train.full, test.full) %>%
  select(Id, Family_Hist_1, Product_Info_1, Product_Info_2)

table(cat.vars$Family_Hist_1, useNA = 'ifany')
table(cat.vars$Product_Info_2, useNA = 'ifany')


cat.vars <- bind_cols(cat.vars[,1], as.data.frame(lapply(cat.vars[,-1], as.factor)))
cat.vars.sparse <- sparse.model.matrix(Id~.-1, cat.vars)
head(cat.vars.sparse)
str(cat.vars.sparse)

cat.vars.sparse <- as.matrix(cat.vars.sparse)

head(cat.vars.sparse)
dim(cat.vars.sparse)

##############################
#Add weights, impute values

features <- train.full %>%
  select(matches("Feature"))

cor(train.full$Feature_1, train.full$Feature_2, use="pairwise.complete.obs")
cor(train.full$Feature_1, train.full$Feature_2, use="complete.obs")
f.cor <- cor(features, use="pairwise.complete.obs")

f.cor <- apply(f.cor, 2, function (x) ifelse(x==1,0,x))
f.cor.max <- apply(f.cor, 1, max)

rets <- train.full %>%
  select(Ret_MinusTwo, Ret_MinusOne, Ret_PlusOne, Ret_PlusTwo)
r.cor <- cor(rets, use="pairwise.complete.obs")

plot(density(rnorm(1000,0,2)))
plot(density(rnorm(1000,0,10)))
plot(density(rcauchy(1000,0,2)))

train_y2 <- train.full %>%
  replace_na(list(Feature_1=0,Feature_2=0,Feature_3=0,Feature_4=0,Feature_5=0,Feature_6=0,Feature_7=0,Feature_8=0,Feature_9=0,Feature_10=0,
                  Feature_11=0,Feature_12=0,Feature_13=0,Feature_14=0,Feature_15=0,Feature_16=0,Feature_17=0,Feature_18=0,Feature_19=0,Feature_20=0,
                  Feature_21=0,Feature_22=0,Feature_23=0,Feature_24=0,Feature_25=0)) %>%
  #filter(!is.na(Feature_2), !is.na(Feature_3)) %>%
  sample_n(5000)

features <- train_y2 %>%
  select(matches("Feature")) %>%
  as.matrix()

dat <- list('N' = dim(train_y2)[[1]],
            'covar1' = train_y2$Feature_2,
            'covar2' = train_y2$Feature_3,
            "y_m2" = train_y2$Ret_MinusTwo,
            "y_m1" = train_y2$Ret_MinusOne,
            'y' = train_y2$Ret_PlusOne,
            'weights' = train_y2$Weight_Daily)

fit <- stan('stan_model_3.stan',  
            model_name = "Stan1", 
            iter=1500, warmup=500,
            thin=2, chains=4, seed=252014,
            data = dat)

print(fit, pars=c("beta", "theta", "sigma"), probs=c(0.5, 0.75, 0.95))
traceplot(fit, pars=c("beta", "theta", 'sigma'))

##############################
#Add weights, impute values

train_y2 <- train.full %>%
  replace_na(list(Feature_1=0,Feature_2=0,Feature_3=0,Feature_4=0,Feature_5=0,Feature_6=0,Feature_7=0,Feature_8=0,Feature_9=0,Feature_10=0,
                  Feature_11=0,Feature_12=0,Feature_13=0,Feature_14=0,Feature_15=0,Feature_16=0,Feature_17=0,Feature_18=0,Feature_19=0,Feature_20=0,
                  Feature_21=0,Feature_22=0,Feature_23=0,Feature_24=0,Feature_25=0)) %>%
  #filter(!is.na(Feature_2), !is.na(Feature_3)) %>%
  sample_n(5000)

features <- train_y2 %>%
  select(matches("Feature")) %>%
  as.matrix()

dat <- list('N' = dim(train_y2)[[1]],
            'covar' = features,
            "y_m2" = train_y2$Ret_MinusTwo,
            "y_m1" = train_y2$Ret_MinusOne,
            'y' = train_y2$Ret_PlusOne,
            'weights' = train_y2$Weight_Daily)

fit <- stan('stan_garch.stan',  
            model_name = "Stan1", 
            iter=1500, warmup=500,
            thin=2, chains=4, seed=252014,
            data = dat)

print(fit, pars=c("beta", "theta", "sigma"), probs=c(0.5, 0.75, 0.95))
traceplot(fit, pars=c("beta", "theta", 'sigma'))

print(fit, pars=c("mu", "sigma"), probs=c(0.5, 0.75, 0.95))

