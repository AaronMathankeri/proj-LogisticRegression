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
nSamples <- 30
xOne <- runif( nSamples, -1, -1) + rnorm( nSamples, 0, 0.2 )
xTwo <- runif( nSamples, 0, 0) + rnorm( nSamples, 0, 0.2 )
xThree <- runif( nSamples, 1, 1) + rnorm( nSamples, 0, 0.2 )
x1 <- c( xOne, xTwo, xThree)
# =============================================================================
# Targets
yOne <- runif( nSamples, -1, -1) + rnorm( nSamples, 0, 0.2 )
yTwo <- runif( nSamples, 0, 0) + rnorm( nSamples, 0, 0.2 )
yThree <- runif( nSamples, 1, 1) + rnorm( nSamples, 0, 0.2 )
x2 <- c( yOne, yTwo, yThree)
# =============================================================================
# assign labels
a <- rep( '0', nSamples)
b <- rep( '1', nSamples)
c <- rep( '2', nSamples)
class <- c( a, b, c )
# =============================================================================
# create dataframe for plots
df <- cbind.data.frame( x1, x2, class )
# =============================================================================
# plot data
ggplot(df, aes(x = x1, y = x2)) +
  geom_point(aes(color=class, shape=class))
# =============================================================================
# =============================================================================
# write to file
#write(inputs, file = "../data/inputs.txt", ncolumns = 1)
#write(targets, file = "../data/targets.txt", ncolumns = 1)

