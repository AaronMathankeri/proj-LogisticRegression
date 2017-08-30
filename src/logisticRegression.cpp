#include "logisticRegression.hpp"

void computeDataMatrix( const double *x1, const double *x2, double *X ){
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    X[i*(ORDER - 1) + 0] = x1[i];
	    X[i*(ORDER - 1) + 1] = x2[i];
      }
}


void computeDesignMatrix( const double *x, double *Phi ){
      double *psi, *ones;
      const int incx = 1;
      
      psi = (double *)mkl_malloc( NUM_PATTERNS*(ORDER - 1)*sizeof( double ), 64 );
      ones = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );

      memset( psi, 0.0, NUM_PATTERNS* (ORDER - 1)*sizeof(double));
      fill_n(ones, NUM_PATTERNS, 1.0); // create ones vector

      computeIdentityBasisFuncs( x, psi );
      
      //set first column to 1--dummy index to calculate w0
      cblas_dcopy(NUM_PATTERNS, ones, incx, Phi, ORDER);
      // set columns 1 ... M-1 with basis function vectors
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    for (int j = 0; j < ORDER; ++j) {
		  if ( j > 0) {
			int p = j - 1;
			Phi[i*ORDER + j] = psi[i*(ORDER-1) + p];			
		  }
	    }
      }
      mkl_free( ones );
      mkl_free( psi );
}

void computeIdentityBasisFuncs( const double *x, double *psi ){
      // psi(x) = x --> identity basis for linear regression
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    for (int j = 0; j < (ORDER - 1); ++j) {
		  psi[i*(ORDER - 1) + j] = x[i*(ORDER-1) + j];			
	    }
      }
}

//---------------------------------------------------------------------------------
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

double computeLeastSquaresError( const double *t, const double *y ){
      double error = 0.0;
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    error += (y[i] - t[i]) * (y[i] - t[i]);
      }
      error *= 0.5;
      return error;
}

void computeMatrixR( const double *y, double *R ){
      // Compute R - matrix
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    R[i*NUM_PATTERNS + i] = y[i] * (1.0 - y[i]);
      }
}

void setRandomWeights( double *weights ){
      for (int i = 0; i < ORDER; ++i) {
	    double temp = fRand( -1.0, 10.0);
	    weights[i] = temp;
      }
}

double fRand( const double fMin, const double fMax){
      double f = (double)rand() / RAND_MAX;
      return fMin + f * (fMax - fMin);
}

//---------------------------------------------------------------------------------
void logisticSigmoid( double &a ){
      // sigma(a) = (1 + exp(-a) )^-1
      a = 1.0/( 1.0 + exp(-a) );
}
//---------------------------------------------------------------------------------
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

