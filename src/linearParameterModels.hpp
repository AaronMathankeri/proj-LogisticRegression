#ifndef LINEARPARAMETERMODELS_H
#define LINEARPARAMETERMODELS_H

#include <vector>
#include <math.h>
#include <stdlib.h>
#include "mkl.h"
#include "parameters.hpp"
#include "ioFunctions.hpp"

void computeDesignMatrix( const double *x, double *Phi, const int form);
void computePseudoInverse( const double* Phi , double *phiPsuedoInverse );
void solveNormalEquations( const double *inversePhi, const double *t, double *w );
void computeOutputs( const double *x, const double *w , double *y );
void computeIdentityBasisFuncs( const double *x , double *psi );
void computePolynomialBasisFuncs( const double *x , double *psi );
double computeLeastSquaresError( const double *t, const double *y );
void setRandomWeights( double *weights );
double fRand( const double fMin, const double fMax);

#endif /* LINEARPARAMETERMODELS_H */
