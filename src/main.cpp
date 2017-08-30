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

void computeGradient( const double *y, const double *t, const double *designMatrix, double *gradE ){
      //gradE = phi' * ( y - t )
      //1. diff <- y -t
      double *diff  = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      memset( diff, 0.0,  NUM_PATTERNS* sizeof(double));
      vdSub( NUM_PATTERNS, y, t, diff);

      //2. phi' * y
      double alpha, beta;
      alpha = 1.0;
      beta = 0.0;
      const int incx = 1;
      cblas_dgemv(CblasRowMajor, CblasTrans, NUM_PATTERNS, ORDER, alpha, designMatrix,
		  ORDER, diff, incx, beta, gradE, incx);

      mkl_free( diff );
}

void computeHessian( const double *designMatrix, const double *R, double *Hessian ){
      //H = phi' * R * phi
      memset( Hessian, 0.0,  ORDER * ORDER* sizeof(double));
      double alpha, beta;
      alpha = 1.0;
      beta = 0.0;
      double *A = (double *)mkl_malloc( ORDER*NUM_PATTERNS*sizeof( double ), 64 );
      memset( A, 0.0,  ORDER * NUM_PATTERNS* sizeof(double));

      //calculate product = Phi' * R = A
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
		  ORDER, NUM_PATTERNS, NUM_PATTERNS, alpha, designMatrix,
		  ORDER, R, NUM_PATTERNS, beta, A, NUM_PATTERNS);
      // A * phi = Hessian
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
		  ORDER, ORDER, NUM_PATTERNS, alpha, A,
		  NUM_PATTERNS, designMatrix, ORDER, beta, Hessian, ORDER);

      mkl_free( A );
}

void computeInverseHessian( double *Hessian ){
      //declare MKL variables for inverse calculation
      int LWORK = ORDER*ORDER;
      int INFO;
      int *IPIV = (int *)mkl_malloc( (ORDER+1)*sizeof( int ), 64 );
      double *WORK = (double *)mkl_malloc( LWORK*sizeof( double ), 64 );

      //calculate inverse A = (A)^-1
      dgetrf( &ORDER, &ORDER, Hessian, &ORDER, IPIV, &INFO );
      dgetri( &ORDER, Hessian, &ORDER, IPIV, WORK, &LWORK, &INFO );

      mkl_free( IPIV );
      mkl_free( WORK );
}

void logisticSigmoid( double &a ){
      // sigma(a) = (1 + exp(-a) )^-1
      a = 1.0/( 1.0 + exp(-a) );
}

double mySigmoid( double a ){
      // sigma(a) = (1 + exp(-a) )^-1
      a = 1.0/( 1.0 + exp(-a) );
      return a;
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
      memset( y, 0.0,  NUM_PATTERNS * sizeof(double));
      cblas_dgemv(CblasRowMajor, CblasNoTrans, NUM_PATTERNS, ORDER, alpha, designMatrix,
		  ORDER, weights, incx, beta, y, incx);

      //now apply to sigmoid to each element
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    logisticSigmoid( y[i] );
      }
}

void computeUpdates( const double *gradE, const double *invHessian, double *deltaWeights ){
      //2. deltaWeights = invHessian * gradE
      double alpha, beta;
      alpha = 1.0;
      beta = 0.0;
      const int incx = 1;
      cblas_dgemv(CblasRowMajor, CblasNoTrans, ORDER, ORDER, alpha, invHessian,
		  ORDER, gradE, incx, beta, deltaWeights, incx);
}

void updateWeights( double *weights, double *deltaWeights ){
      vdSub( ORDER, weights, deltaWeights, weights);
}

int main(int argc, char *argv[])
{
      cout << " Aaron's Back." << endl;

      //--------------------------------------------------------------------------------
      // declare variables for calculations
      double *x1, *x2, *t, *X; //data
      double *weights, *y, *designMatrix, *R; //logistic regression parameters

      double *gradE, *Hessian, *deltaWeights;
      
      x1 = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      x2 = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      t = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      X = (double *)mkl_malloc( NUM_PATTERNS*(ORDER - 1)*sizeof( double ), 64 );
      
      weights = (double *)mkl_malloc( ORDER*sizeof( double ), 64 );
      y = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      designMatrix = (double *)mkl_malloc( NUM_PATTERNS*ORDER*sizeof( double ), 64 );
      R = (double *)mkl_malloc( NUM_PATTERNS*NUM_PATTERNS*sizeof( double ), 64 );
      
      gradE = (double *)mkl_malloc( ORDER*sizeof( double ), 64 );
      Hessian = (double *)mkl_malloc( ORDER*ORDER*sizeof( double ), 64 );
      deltaWeights = (double *)mkl_malloc( ORDER*sizeof( double ), 64 );
      
      memset( x1, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( x2, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( t, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( X, 0.0,  NUM_PATTERNS *(ORDER - 1)* sizeof(double));

      memset( weights, 0.0,  ORDER * sizeof(double));
      memset( y, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( designMatrix, 0.0,  NUM_PATTERNS * ORDER* sizeof(double));
      memset( R, 0.0,  NUM_PATTERNS *NUM_PATTERNS* sizeof(double));

      memset( gradE, 0.0, ORDER * sizeof(double));
      memset( Hessian, 0.0,  ORDER * ORDER* sizeof(double));
      memset( deltaWeights, 0.0, ORDER * sizeof(double));
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
      printFeatures( x1, x2, 5 );
      
      //cout << "\nFirst 10 Targets" << endl;
      //printVector( t, 10 );
      //--------------------------------------------------------------------------------
      //1. Randomly initialize weights.( srandnull() ) for true randomization
      setRandomWeights( weights );
      cout << "\nInitial Weights" << endl;
      printVector( weights, ORDER );
      //--------------------------------------------------------------------------------
      //2. Compute outputs
      // put all data into X matrix!
      computeDataMatrix( x1, x2, X );
      //cout << "\nData matrix" << endl;
      //printMatrix( X, NUM_PATTERNS, 2 );
      //--------------------------------------------------------------------------------
      // design matrix is just X with a column of ones at the first position
      computeDesignMatrix( X, designMatrix );
      cout << "\nComputing Design matrix with identity basis functions..." << endl;
      //printMatrix( designMatrix, NUM_PATTERNS, ORDER);
      //--------------------------------------------------------------------------------
      // y = sigma( Phi'*w)
      computeMyOutputs( weights, designMatrix, y );
      //cout << "\nFirst 10 Outputs" << endl;
      //printVector( y, 100 );
      cout << "\n\nInitial error is " << computeLeastSquaresError( t, y ) << endl;
      //--------------------------------------------------------------------------------
      // Compute R - matrix
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    R[i*NUM_PATTERNS + i] = y[i] * (1.0 - y[i]);
      }

      /*
      cout << "R matrix" << endl;
      for (int i=0; i < 10; i++) {
	    for (int j=0; j < 10; j++) {
		  printf ("%12.5f", R[i*NUM_PATTERNS +j]);
	    }
	    printf ("\n");
      }
      */

      //--------------------------------------------------------------------------------
      computeGradient( y , t, designMatrix, gradE );
      cout << "\nGradient :" <<endl;
      printVector( gradE, ORDER );
      //--------------------------------------------------------------------------------
      computeHessian( designMatrix, R, Hessian );
      cout << "\nHessian :" <<endl;
      printMatrix( Hessian, ORDER, ORDER );
      //--------------------------------------------------------------------------------
      computeInverseHessian( Hessian );
      cout << "\nInverse Hessian :" <<endl;
      printMatrix( Hessian, ORDER, ORDER );
      //--------------------------------------------------------------------------------
      computeUpdates( gradE, Hessian, deltaWeights );
      cout << "\nChange in weights is :" << endl;
      printVector( deltaWeights, ORDER );
      //--------------------------------------------------------------------------------
      updateWeights( weights, deltaWeights );
      cout << "\nNew Weights" << endl;
      printVector( weights, ORDER );

      cout << "\n\nNew error is " << computeLeastSquaresError( t, y ) << endl;
      computeMyOutputs( weights, designMatrix, y );
      //cout << "\nFirst 10 Outputs" <<endl;
      //printVector( y, 100 );
      cout << "\n\nNew error is " << computeLeastSquaresError( t, y ) << endl;

      //--------------------------------------------------------------------------------
      // Newton's method!
      //1. compute gradient
      //2. compute Hessian
      //3. compute Inverse Hessian
      //4. compute update
      //5. apply update
      //--------------------------------------------------------------------------------
      cout << "\nBEGIN OPTIMIZATION VIA NEWTON'S METHOD" << endl;
      double thresh = 1e-3;
      double oldError = computeLeastSquaresError( t , y);
      double newError = oldError + 1e2;
      //cout << fabs(newError - oldError) << endl;

      while (fabs(newError - oldError) > thresh ) {
	    computeMyOutputs( weights, designMatrix, y );
	    oldError = computeLeastSquaresError( t , y );
	    //--------------------------------------------------------------------------------
	    // Compute R - matrix
	    for (int i = 0; i < NUM_PATTERNS; ++i) {
		  R[i*NUM_PATTERNS + i] = y[i] * (1.0 - y[i]);
	    }
	    //--------------------------------------------------------------------------------
	    computeGradient( y , t, designMatrix, gradE );
	    //--------------------------------------------------------------------------------
	    computeHessian( designMatrix, R, Hessian );
	    //--------------------------------------------------------------------------------
	    computeInverseHessian( Hessian );
	    //--------------------------------------------------------------------------------
	    computeUpdates( gradE, Hessian, deltaWeights );
	    //--------------------------------------------------------------------------------
	    updateWeights( weights, deltaWeights );
	    //--------------------------------------------------------------------------------
	    computeMyOutputs( weights, designMatrix, y );
	    newError = computeLeastSquaresError( t , y );
	    cout << "\n\nNew error is " << computeLeastSquaresError( t, y ) << endl;
      }
      
      cout << "\nOptimal Weights" << endl;
      printVector( weights, ORDER );
      //--------------------------------------------------------------------------------

      printf ("\n Deallocating memory \n\n");
      mkl_free( x1 );
      mkl_free( x2 );
      mkl_free( t );
      mkl_free( weights );
      mkl_free( y );
      mkl_free( designMatrix );
      mkl_free( R );
      mkl_free( X );
      mkl_free( gradE );
      mkl_free( Hessian );
      printf (" Example completed. \n\n");

      return 0;
}
