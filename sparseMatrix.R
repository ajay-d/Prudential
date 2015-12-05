set.seed(123)
n <- 6
df <- data.frame(x = sample(c("A", "B", "C"), n, TRUE),
                 y = sample(c("D", "E"),      n, TRUE), stringsAsFactors=T)

library(Matrix)
sparse.model.matrix(~.-1,data=df)
as.matrix(sparse.model.matrix(~.-1,data=df))

n <- nrow(df)
nlevels <- sapply(df, nlevels)
i <- rep(seq_len(n), ncol(df))
j <- unlist(lapply(df, as.integer)) +
  rep(cumsum(c(0, head(nlevels, -1))), each = n)
x <- 1
sparseMatrix(i = i, j = j, x = x)
m <- as.matrix(sparseMatrix(i = i, j = j, x = x))

#https://github.com/dmlc/xgboost/blob/master/R-package/vignettes/discoverYourData.Rmd
#https://github.com/dmlc/xgboost/blob/master/R-package/vignettes/xgboostPresentation.Rmd


#https://github.com/dmlc/xgboost/blob/master/demo/kaggle-otto/otto_train_pred.R
#https://github.com/dmlc/xgboost/blob/master/R-package/demo/basic_walkthrough.R

t <- train.full %>%
  select(Id, Product_Info_5) %>%
  mutate(p5 = R.utils::intToBin(Product_Info_5)) %>%
  mutate(p5.num = as.numeric(p5)) %>%
  separate(p5.num, c('A', 'B'), sep=1, convert=TRUE)

all.cat.vars <- train.full %>%
  select(Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, 
         Employment_Info_2, Employment_Info_3, Employment_Info_5, 
         InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, 
         Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, 
         Family_Hist_1, 
         Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_10, 
         Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, 
         Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, 
         Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, 
         Medical_History_41)
length(names(all.cat.vars))

library(lazyeval)
library(R.utils)
cat.table <- NULL
for(i in names(all.cat.vars)) {
  
  #print(all.cat.vars %>% count_(i))
  
  dots <- list(interp(~ifelse(is.na(var), 1, 0), var = as.name(paste0(i))))
  
  #Count number of NA columns
  na.cnt <- all.cat.vars %>% 
    mutate_(.dots = setNames(dots, paste("na.var"))) %>% 
    count(na.var)
  
  #Count cardinality
  df <- data_frame(var = i,
                   categories = nrow(all.cat.vars %>% count_(i)),
                   populated = na.cnt %>% filter(na.var==0) %>% select(n) %>% as.numeric)
  
  cat.table <- bind_rows(cat.table, df)
}

train.full %>%
  mutate(na.var = ifelse(is.na(Employment_Info_1), 1, 0)) %>%
  count(na.var)

train.full %>% count(Employment_Info_2)
train.full %>% count(Product_Info_2)
train.full %>% count(Product_Info_3)

train.full %>% summarise(max(Employment_Info_2))

paste0(intToBin(30), collapse = ".")
paste0(strsplit(intToBin(30), split='')[[1]], collapse = ".")

##Employment_Info_2
train.full %>%
  select(Id, Employment_Info_2) %>%
  mutate(bin = intToBin(Employment_Info_2)) %>%
  rowwise() %>%
  mutate(bin.split = paste0(strsplit(bin, split='')[[1]], collapse = ".")) %>%
  separate(bin.split, LETTERS[1:6], convert=TRUE)


col.names <- paste("Employment_Info_2", 1:6, sep='.')

train.full %>%
  select(Id, Employment_Info_2) %>%
  mutate(bin = intToBin(Employment_Info_2)) %>%
  rowwise() %>%
  mutate(bin.split = paste0(strsplit(bin, split='')[[1]], collapse = ".")) %>%
  separate(bin.split, col.names, convert=TRUE)

##Product_Info_2
prod.2 <- train.full %>% 
  count(Product_Info_2) %>%
  mutate(prod=row_number())

intToBin(19)
col.names <- paste("Product_Info_2", 1:5, sep='.')

train.full %>%
  left_join(prod.2, by='Product_Info_2') %>%
  select(Id, prod) %>%
  mutate(bin = intToBin(prod)) %>%
  rowwise() %>%
  mutate(bin.split = paste0(strsplit(bin, split='')[[1]], collapse = ".")) %>%
  separate(bin.split, col.names, convert=TRUE)


evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(mean(abs(labels - preds)))
  print(err)
  return(list(metric = "error", value = err))
}

######################################################

set.seed(1)

train.sample <- train.full %>%
  select(2:12, -Product_Info_2, matches("InsuredInfo"), Response) %>%
  sample_n(1000)

y <- train.sample %>%
  select(Response) %>%
  mutate(Response=Response-1) %>%
  as.matrix

train.sample <- train.sample %>%
  select(-Response) %>%
  as.matrix

test.sample <- train.full %>%
  select(2:12, -Product_Info_2, matches("InsuredInfo"), Response) %>%
  sample_n(1000)

test.y <- test.sample %>%
  select(Response) %>%
  as.matrix

test.sample <- test.sample %>%
  select(-Response) %>%
  as.matrix

dtrain.sample <- xgb.DMatrix(data = train.sample, label = y)

param <- list("objective" = "multi:softmax",
              "eval_metric" = "mlogloss",
              "num_class" = 8, "nthread" = 8)
bst.1 <- xgb.train(param = param, data = dtrain, nrounds = nround, verbose = 2)

pred <- predict(bst.1, test.sample)
#length(pred)/8
length(pred)
head(pred)
head(test.y)

#http://www.real-statistics.com/reliability/cohens-kappa/
#http://www.real-statistics.com/reliability/weighted-cohens-kappa/

scores <- cbind(test.y, pred+1) %>%
  as.data.frame %>%
  setNames(c('actual', 'predicted'))

scores <- data_frame('actual' = c(1,4,2,2,5,2,5,4,5,3,1),
                     'predicted' = c(3,4,4,4,4,4,6,7,8,9,10))

scores %>%
  count(actual, predicted) %>%
  spread(predicted, n)

#histogram
hist <- scores %>%
  count(actual, predicted) %>%
  ungroup %>%
  complete(actual, predicted, fill=list(n=0)) %>%
  spread(predicted, n) %>%
  select(-actual)

hist

#weights
weights <- scores %>%
  count(actual, predicted) %>%
  ungroup %>%
  complete(actual, predicted, fill=list(n=0)) %>%
  mutate(N = n(),
         w = (actual-predicted)^2 / (7-1)^2) %>%
  select(-N, -n) %>%
  spread(predicted, w) %>%
  select(-actual)

weights

#expected
expected <- scores %>%
  count(actual, predicted) %>%
  ungroup %>%
  complete(actual, predicted, fill=list(n=0)) %>%
  group_by(predicted) %>%
  mutate(predicted.total = sum(n)) %>%
  group_by(actual) %>%
  mutate(actual.total = sum(n)) %>%
  ungroup() %>%
  mutate(N = sum(n),
         predicted.exp = predicted.total/N,
         actual.exp = actual.total/N,
         expected = predicted.exp * actual.exp,
         expected.n = predicted.exp * actual.exp * N) %>%
  select(actual, predicted, expected.n) %>%
  spread(predicted, expected.n) %>%
  select(-actual)

expected

num <- sum(as.matrix(weights) * as.matrix(hist))
den <- sum(as.matrix(weights) * as.matrix(expected))
1-num/den
