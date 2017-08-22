//Copyright (C) 2017 Aaron Mathankeri (aaron@quantifiedag.com)
//License: QuantifiedAG Software License.  See LICENSE.txt for full license.

/*
 *   main.cpp
 *   \brief HPC implementation of logistic regression
 *
 *  p(c=1|x) = sigma( x*w )
 * 
 */
#include <iostream>
#include "mkl.h"
#include "ioFunctions.hpp"
#include "parameters.hpp"
#include "logisticRegression.hpp"

using namespace std;

void computeGradient( double *x, double *t, double *designMatrix ){
      //gradE = phi' * ( y - t )
      ;
}

void computeHessian( double *x, double * designMatrix ){
      //H = phi' * R * phi
      ;
}
/*
void computeOutputs( double *x, double *w, double *y ){
      // y = sigma( w'* x )
      ;
}
*/
void logisticSigmoid( double &a ){
      // sigma(a) = (1 + exp(-a) )^-1
      a = 1.0/( 1 + exp(-a) );
}

void computeDataMatrix( const double *x1, const double *x2, double *X ){
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    X[i*(ORDER - 1) + 0] = x1[i];
	    X[i*(ORDER - 1) + 1] = x2[i];
      }
}
void computeMyOutputs( const double *weights, const double *designMatrix, double *y ){
      // y = sigma( Phi'*w)
      double alpha, beta;
      alpha = 1.0;
      beta = 0.0;
      const int incx = 1;

      cblas_dgemv(CblasRowMajor, CblasNoTrans, NUM_PATTERNS, ORDER, alpha, designMatrix,
		  ORDER, weights, incx, beta, y, incx);

      //now apply to sigmoid to each element
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    logisticSigmoid( y[i] );
      }
}
int main(int argc, char *argv[])
{
      cout << " Aaron's Back." << endl;

      //--------------------------------------------------------------------------------
      // declare variables for calculations
      double *x1, *x2, *t, *X; //data
      double *weights, *y, *designMatrix, *R, *z; //logistic regression parameters

      x1 = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      x2 = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      t = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      X = (double *)mkl_malloc( NUM_PATTERNS*(ORDER - 1)*sizeof( double ), 64 );
      
      weights = (double *)mkl_malloc( ORDER*sizeof( double ), 64 );
      y = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      designMatrix = (double *)mkl_malloc( NUM_PATTERNS*ORDER*sizeof( double ), 64 );
      R = (double *)mkl_malloc( NUM_PATTERNS*NUM_PATTERNS*sizeof( double ), 64 );
      z = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      

      memset( x1, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( x2, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( t, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( X, 0.0,  NUM_PATTERNS *(ORDER - 1)* sizeof(double));

      memset( weights, 0.0,  ORDER * sizeof(double));
      memset( y, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( designMatrix, 0.0,  NUM_PATTERNS * ORDER* sizeof(double));
      memset( R, 0.0,  NUM_PATTERNS *NUM_PATTERNS* sizeof(double));
      memset( z, 0.0,  NUM_PATTERNS * sizeof(double));
      //--------------------------------------------------------------------------------
      //read data
      string x1File = "./data/train/x1.txt";
      string x2File = "./data/train/x2.txt";
      string targetsFile = "./data/train/class.txt";

      loadData( x1 , x1File );
      loadData( x2 , x2File );
      loadData( t , targetsFile );
      
      cout << "First 10 Features" << endl;
      cout << "x1\t | \tx2" << endl;
      printFeatures( x1, x2, 10 );
      
      cout << "\nFirst 10 Targets" << endl;
      printVector( t, 10 );
      //--------------------------------------------------------------------------------
      //1. Randomly initialize weights.( srandnull() ) for true randomization
      setRandomWeights( weights );
      cout << "\nInitial Weights" << endl;
      printVector( weights, ORDER );
      //--------------------------------------------------------------------------------
      //2. Compute outputs
      // put all data into X matrix!
      computeDataMatrix( x1, x2, X );
      //--------------------------------------------------------------------------------
      // design matrix is just X with a column of ones at the first position
      computeDesignMatrix( X, designMatrix );
      cout << "\nDesign matrix" << endl;
      printMatrix( designMatrix, 10, 3);
      //--------------------------------------------------------------------------------
      // y = sigma( Phi'*w)
      computeMyOutputs( weights, designMatrix, y );
      cout << "\n First 10 Outputs" <<endl;
      printVector( y, 10 );
      //--------------------------------------------------------------------------------
      // Compute R - matrix
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    R[i*NUM_PATTERNS + i] = y[i] * (1.0 - y[i]);
      }

      cout <<"Top right corner of R matrix" << endl;
      for (int i=0; i < 10; i++) {
	    for (int j=0; j < 10; j++) {
		  printf ("%12.5f", R[i*NUM_PATTERNS +j]);
	    }
	    printf ("\n");
      }
      //--------------------------------------------------------------------------------

      //--------------------------------------------------------------------------------
      //--------------------------------------------------------------------------------
      printf ("\n Deallocating memory \n\n");
      mkl_free( x1 );
      mkl_free( x2 );
      mkl_free( t );
      mkl_free( weights );
      mkl_free( y );
      mkl_free( designMatrix );
      mkl_free( R );
      mkl_free( z );
      mkl_free( X );
      printf (" Example completed. \n\n");

      return 0;
}
