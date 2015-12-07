evalkappa <- function(actual, preds) {

  scores <- cbind(actual, preds) %>%
    as.data.frame %>%
    setNames(c('actual', 'predicted'))
  
  #histogram
  hist <- scores %>%
    count(actual, predicted) %>%
    ungroup %>%
    complete(actual, predicted, fill=list(n=0)) %>%
    spread(predicted, n) %>%
    select(-actual)
  
  #hist
  
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
  
  #weights
  
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
  
  #expected
  
  num <- sum(as.matrix(weights) * as.matrix(hist))
  den <- sum(as.matrix(weights) * as.matrix(expected))
  err <- 1-num/den
  
  return(list(metric = "Quad Kappa", value = err))
}

##function for 
evalkappa.gbm <- function(actual, dtrain) {
  
  preds <- getinfo(dtrain, "label") + 1
  
  scores <- cbind(actual + 1, preds) %>%
    as.data.frame %>%
    setNames(c('actual', 'predicted'))
  
  #histogram
  hist <- scores %>%
    count(actual, predicted) %>%
    ungroup %>%
    complete(actual, predicted, fill=list(n=0)) %>%
    spread(predicted, n) %>%
    select(-actual)
  
  #hist
  
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
  
  #weights
  
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
  
  #expected
  
  num <- sum(as.matrix(weights) * as.matrix(hist))
  den <- sum(as.matrix(weights) * as.matrix(expected))
  err <- 1-num/den
  
  return(list(metric = "Quad Kappa", value = err))
}
