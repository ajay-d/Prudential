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

set.seed(2016)

train.full <- read_csv("data/train.csv.zip")
test.full <- read_csv("data/test.csv.zip")
sample_submission <- read_csv("data/sample_submission.csv.zip")

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
    mutate_(.dots = setNames(dots, paste(dummy.var.name))) %>%
    select_(interp(~-(x), x=as.name(paste0(i))))
  
}





