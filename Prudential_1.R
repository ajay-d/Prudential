library(readr)
library(dplyr)
library(tidyr)
library(rstan)
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
  select(matches("^Ret_")) %>%
  names

train.full %>%
  select(matches("Feature")) %>%
  names

ggplot(train.full) + geom_histogram(aes(Ret_MinusTwo), binwidth=.01)
ggplot(train.full) + geom_histogram(aes(Ret_MinusOne), binwidth=.01)

summary(train.full$Ret_MinusTwo)
summary(train.full$Ret_MinusOne)
summary(train.full$Weight_Daily)

train.full %>%
  mutate(na.var = ifelse(is.na(Ret_PlusOne), 1, 0)) %>%
  count(na.var)

train.full %>%
  mutate(na.var = ifelse(is.na(Feature_2), 1, 0)) %>%
  count(na.var)

train.full %>%
  mutate(na.var = ifelse(is.na(Weight_Daily), 1, 0)) %>%
  count(na.var)

ggplot(train.full) + geom_histogram(aes(Weight_Daily))

ggplot(train.full) + geom_histogram(aes(Feature_2), binwidth=.01)
ggplot(train.full) + geom_histogram(aes(Feature_3), binwidth=.01)
ggplot(train.full) + geom_histogram(aes(Feature_4), binwidth=.01)

stan.model <- "
  data {
    int<lower=0> N;
    vector [N] y;
    vector [N] covar1;
    vector [N] covar2;
    vector [N] y_m2;
    vector [N] y_m1;

    //one-dimensional array of size N containing real values
    //real y[N];
    //real y_hat[N];
    //vector[N] y_hat;
  }
  parameters {
    //intercept
    real alpha;

    //regression
    real<lower=-1,upper=1> beta1;
    real<lower=-1,upper=1> beta2;

    //moving average
    real<lower=0,upper=1> theta1;

    real<lower=0> sigma;
  }
  transformed parameters {
    real<lower=0,upper=1> theta2;
    theta2 <- 1-theta1;
  }
  model {
    for (n in 1:N)
      y[n] ~ normal(alpha + 
                    beta1*covar1[n] + beta2*covar2[n] + 
                    theta1*y_m1[n] + theta2*y_m2[n],
                    sigma);
  }
"

train_y2 <- filter(train.full, !is.na(Feature_2), 
                   !is.na(Feature_3))

dat <- list('N' = dim(train_y2)[[1]],
            'covar1' = train_y2$Feature_2,
            'covar2' = train_y2$Feature_3,
            "y_m2" = train_y2$Ret_MinusTwo,
            "y_m1" = train_y2$Ret_MinusOne,
            'y' = train_y2$Ret_PlusOne)

fit <- stan(model_code = stan.model, 
            model_name = "Stan1", 
            iter=2500, warmup=500,
            thin=2, chains=4, seed=252014,
            data = dat)


b <- extract(fit, "beta1")$beta1
print(fit, pars=c("beta1", "beta2", "theta1", "theta2", 'sigma'),
      probs=c(0.5, 0.75, 0.95))

traceplot(fit, pars=c("beta1", "beta2", "theta1", "theta2", 'sigma'))
stan_dens(fit, pars=c("beta1", "beta2", "theta1", "theta2"), 
          fill="skyblue")

##############################
#Use model file

fit <- stan('stan_model_3.stan', 
            model_name = "Stan1", 
            iter=1500, warmup=500,
            thin=2, chains=4, seed=252014,
            data = dat)
print(fit, pars=c("beta1", "beta2", "theta"),
      probs=c(0.5, 0.75, 0.95))
traceplot(fit, pars=c("beta1", "beta2", "theta"))

get_elapsed_time(fit)
get_posterior_mean(fit, pars=c("beta1", "beta2", "theta"))

#gg objects
stan_plot(fit)
stan_trace(fit)

stan_plot(fit, show_density=TRUE, point_est='median')

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

