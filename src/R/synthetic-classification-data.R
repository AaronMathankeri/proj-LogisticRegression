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
nSamples <- 10
xOne <- runif( nSamples, -1, -1) + rnorm( nSamples, 0, 0.1 )
xTwo <- runif( nSamples, 0, 0) + rnorm( nSamples, 0, 0.1 )
x1 <- c( xOne, xTwo )
# =============================================================================
# Targets
yOne <- runif( nSamples, -1, -1) + rnorm( nSamples, 0, 0.1 )
yTwo <- runif( nSamples, 0, 0) + rnorm( nSamples, 0, 0.1 )
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
  geom_point(aes(color=class, shape=class))
# =============================================================================
# logistic regression in R
#model <- glm()
# =============================================================================
# write to file
#write(inputs, file = "../data/inputs.txt", ncolumns = 1)
#write(targets, file = "../data/targets.txt", ncolumns = 1)

