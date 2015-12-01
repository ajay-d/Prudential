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
