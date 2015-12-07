rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(Matrix)
library(R.utils)
library(xgboost)
library(ggplot2)
library(lazyeval)

options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10)

train.full <- read_csv("data/train.csv.zip")
test.full <- read_csv("data/test.csv.zip")
sample_submission <- read_csv("data/sample_submission.csv.zip")

source("kappa_functions.R")

#Combine test and train
#convert categorical vars
#impute missing variables

combined.data <- train.full %>%
  select(-Response) %>%
  bind_rows(test.full)

#check columns for NAs
combined.data %>%
  select(1:12, matches("Product_Info"), matches("InsuredInfo"), matches("Medical_Keyword"),
         Family_Hist_1, matches("Insurance_History"), -Insurance_History_5,
         Employment_Info_2, Employment_Info_3, Employment_Info_5) %>%
  complete.cases %>%
  sum

all.cat.vars <- combined.data %>%
  select(Id,
         Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, 
         Employment_Info_2, Employment_Info_3, Employment_Info_5,
         InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, 
         Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9,
         Family_Hist_1)

cat.table <- NULL
for(i in names(all.cat.vars)) {

  #print(all.cat.vars %>% count_(i))
  
  #get highest category, and count
  var.max <- all.cat.vars %>%
    count_(i, sort=TRUE) %>%
    filter(row_number()==1)
  
  dots <- list(interp(~ifelse(is.na(var), 1, 0), var = as.name(paste0(i))))
  
  #Count number of NA columns
  na.cnt <- all.cat.vars %>% 
    mutate_(.dots = setNames(dots, paste("na.var"))) %>% 
    count(na.var)
  
  #Count cardinality
  df <- data_frame(var = i,
                   categories = nrow(all.cat.vars %>% count_(i)),
                   populated = na.cnt %>% filter(na.var==0) %>% select(n) %>% as.numeric,
                   max.cat.value = as.character(var.max[[1,1]]),
                   max.cat.population = var.max[[1,2]])
  
  cat.table <- bind_rows(cat.table, df)
}
cat.table <- cat.table %>%
  arrange(desc(categories))

ggplot(train.full) + geom_histogram(aes(Response), binwidth=.5)

#code all vars with 2 categories
cat.table.2 <- cat.table %>%
  filter(categories==2)

for(i in cat.table.2$var) {
  
  cur.var.value <- cat.table.2 %>% 
    filter(var==i)

  cur.var <- all.cat.vars %>%
    select_('Id', i)
  
  dummy.var.name <- paste(i, cur.var.value[[1,"max.cat.value"]], sep='.')
  dots <- list(interp(~as.integer(var==cur.var.value[[1,"max.cat.value"]]), 
                      var = as.name(paste0(i))))
               
  all.cat.vars <- all.cat.vars %>%
    mutate_(.dots = setNames(dots, paste(dummy.var.name)))
    #select_(interp(~-(x), x=as.name(paste0(i))))
  
}

#code all vars with 3 categories
cat.table.3 <- cat.table %>%
  filter(categories==3)

for(i in cat.table.3$var) {
  
  cur.var.value <- cat.table.3 %>% 
    filter(var==i)
  
  var.max <- all.cat.vars %>%
    count_(i, sort=TRUE) %>%
    filter(row_number() <= 2)
  
  dummy.var.name.1 <- paste(i, var.max[[1,1]], sep='.')
  dummy.var.name.2 <- paste(i, var.max[[2,1]], sep='.')
  
  dots.1 <- list(interp(~as.integer(var==var.max[[1,1]]), 
                        var = as.name(paste0(i))))
  dots.2 <- list(interp(~as.integer(var==var.max[[2,1]]), 
                        var = as.name(paste0(i))))
  
  all.cat.vars <- all.cat.vars %>%
    mutate_(.dots = setNames(dots.1, paste(dummy.var.name.1))) %>%
    mutate_(.dots = setNames(dots.2, paste(dummy.var.name.2)))
  
}

#Remove old variables
columns.to.drop <- cat.table %>%
  filter(categories==2 | categories==3)
columns.to.drop <- columns.to.drop$var

for(i in columns.to.drop) {
  all.cat.vars <- all.cat.vars %>%
    select_(interp(~-(x), x=as.name(paste0(i))))
}

####build model
train.data <- train.full %>%
  select(Id, Response,
         Product_Info_4, Ins_Age, Ht, Wt, BMI) %>%
  left_join(all.cat.vars) %>%
  select(-Product_Info_3, -Employment_Info_2, -Product_Info_2, -InsuredInfo_3)

set.seed(1)

train.data.model <- train.data %>%
  sample_frac(.75) %>%
  select(-Response)

ids.in.model <- train.data.model$Id

train.data.model.y <- train.data.model %>%
  left_join(train.data, by='Id') %>%
  select(Response) %>%
  mutate(Response=Response-1) %>%
  as.matrix

train.data.watch <- train.data.model %>%
  sample_frac(.25)

train.data.test <- train.data %>%
  anti_join(train.data.model, by='Id') %>%
  select(-Id) %>%
  as.matrix

train.data.model <- train.data.model %>%
  select(-Id) %>%
  as.matrix

train.data.watch.y <- train.data.watch %>%
  left_join(train.data, by='Id') %>%
  select(Response) %>%
  mutate(Response=Response-1) %>%
  as.matrix

train.data.watch <- train.data.watch %>%
  select(-Id) %>%
  as.matrix

dtrain <- xgb.DMatrix(data = train.data.model, label = train.data.model.y)
dtest <- xgb.DMatrix(data = train.data.watch, label = train.data.watch.y)

param.1 <- list("objective" = "multi:softmax",
                #"eval_metric" = "mlogloss",
                "num_class" = 8,
                "max_depth" = 6)

watchlist <- list(train=dtrain, test=dtest)

nround <- 25

bst.1 <- xgb.train(param = param.1, data = dtrain, 
                   watchlist = watchlist,
                   nrounds = nround, verbose = 1,
                   feval = evalkappa.gbm)

param.2 <- list("objective" = "multi:softmax",
                "eval_metric" = "mlogloss",
                "num_class" = 8,
                "max_depth" = 10)

bst.2 <- xgb.train(param = param.2, data = dtrain, 
                   nrounds = nround, verbose = 1)

pred.1 <- predict(bst.1, train.data.test)
pred.2 <- predict(bst.2, train.data.test)

#Add one
pred.1 <- pred.1 + 1
pred.2 <- pred.2 + 1

ggplot(as.data.frame(pred.1)) + geom_histogram(aes(pred.1), binwidth=.5)
ggplot(as.data.frame(pred.2)) + geom_histogram(aes(pred.2), binwidth=.5)

test.data.actual <- train.data %>%
  anti_join(train.data %>%
              filter(Id %in% ids.in.model), by='Id') %>%
  select(Response) %>%
  as.matrix

train.data.actual <- train.data %>%
  filter(Id %in% ids.in.model) %>%
  select(Response) %>%
  as.matrix

getinfo(dtrain, "label") + 1

pred.train <- predict(bst.1, train.data.model)
pred.train <- pred.train + 1 

#training set
evalkappa(train.data.actual, pred.train)
evalkappa(train.data.actual, getinfo(dtrain, "label") + 1)

#test set
evalkappa(test.data.actual, pred.2)

