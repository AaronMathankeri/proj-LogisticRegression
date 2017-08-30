#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <vector>
#include <math.h>
#include <stdlib.h>
#include "mkl.h"
#include "parameters.hpp"
#include "ioFunctions.hpp"


void computeDataMatrix( const double *x1, const double *x2, double *X );
void computeDesignMatrix( const double *x, double *Phi );
void computeIdentityBasisFuncs( const double *x , double *psi );


void computeMyOutputs( const double *weights, const double *designMatrix, double *y );
double computeLeastSquaresError( const double *t, const double *y );
void computeMatrixR( const double *y, double *R );
void setRandomWeights( double *weights );
double fRand( const double fMin, const double fMax);

void logisticSigmoid( double &a );

//newton's method
void computeGradient( const double *y, const double *t, const double *designMatrix, double *gradE );
void computeHessian( const double *designMatrix, const double *R, double *Hessian );
void computeInverseHessian( double *Hessian );
void computeUpdates( const double *gradE, const double *invHessian, double *deltaWeights );
void updateWeights( double *weights, double *deltaWeights );


#endif /* LOGISTICREGRESSION_H */
