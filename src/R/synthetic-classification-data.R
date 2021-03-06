# =============================================================================
# Clear workspace
rm(list=ls())

set.seed(123)
# =============================================================================
setwd('/Users/aaron/playground/proj-LogisticRegression/data/')
# =============================================================================
# libraries
library(ggplot2)
# =============================================================================
# sample from uniform distribution
# x ~ U(-1,1)
nSamples <- 50
xOne <- runif( nSamples, 0, 1) + rnorm( nSamples, 0, 0.1 )
xTwo <- runif( nSamples, 0, 0) + rnorm( nSamples, 0, 0.1 )
x1 <- c( xOne, xTwo )
# =============================================================================
# Targets
yOne <- runif( nSamples, -1, 0) + rnorm( nSamples, 0, 0.1 )
yTwo <- runif(nSamples, 0, 0) + rnorm( nSamples, 0, 0.1 )
x2 <- c( yOne, yTwo )
# =============================================================================
# assign labels
a <- rep( '0', nSamples)
b <- rep( '1', nSamples)
class <- c( a, b )
# =============================================================================
# create dataframe for plots
df <- cbind.data.frame( x1, x2, class )
# =============================================================================
# plot data
ggplot(df, aes(x = x1, y = x2)) +
  geom_point(aes(color=class, shape=class)) + ggtitle("Training Data")
# =============================================================================
# logistic regression in R
model <- glm( class ~ ., family = binomial(link='logit'), data=df)
summary( model )
# =============================================================================
# test
zOne <- runif( nSamples, 0, 1) + rnorm( nSamples, 0, 0.2 )
zTwo <- runif( nSamples, 0, 0) + rnorm( nSamples, 0, 0.2 )
z1 <- c( xOne, xTwo )
wOne <- runif( nSamples, -1, 0) + rnorm( nSamples, 0, 0.2 )
wTwo <- runif( nSamples, 0, 0) + rnorm( nSamples, 0, 0.2 )
z2 <- c( wOne, wTwo )

newDf <- cbind.data.frame( z1, z2, class)
predVals <- predict( model, newDf, type="response")
predictions <- data.frame(z1,z2,class,predVals)

# plot data
ggplot(df, aes(x = z1, y = z2)) +
  geom_point(aes(color=class, shape=class)) + ggtitle("Testing Data")
# # =============================================================================
# write to file
write(x1, file = "../data/train/x1.txt", ncolumns = 1)
write(x2, file = "../data/train/x2.txt", ncolumns = 1)
write(class, file = "../data/train/class.txt", ncolumns = 1)

write(z1, file = "../data/test/z1.txt", ncolumns = 1)
write(z2, file = "../data/test/z2.txt", ncolumns = 1)
