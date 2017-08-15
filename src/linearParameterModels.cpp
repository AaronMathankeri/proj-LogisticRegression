#include "linearParameterModels.hpp"

void computeIdentityBasisFuncs( const double *x, double *psi ){
      // psi(x) = x --> identity basis for linear regression
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    for (int j = 0; j < (ORDER - 1); ++j) {
		  psi[i*(ORDER - 1) + j] = x[i*(ORDER-1) + j];			
	    }
      }
}

void computePolynomialBasisFuncs( const double *x , double *psi ){
      // psi(x) = (x x^2 x^3) --> polynomial basis
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    for (int j = 0; j < (ORDER - 1); ++j) {
		  int p = j + 1;
		  psi[i*(ORDER - 1) + j] = pow(x[i], p);
	    }
      }
}

void computeDesignMatrix( const double *x, double *Phi, const int form ){
      double *psi, *ones;
      const int incx = 1;
      
      psi = (double *)mkl_malloc( NUM_PATTERNS*(ORDER - 1)*sizeof( double ), 64 );
      ones = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );

      memset( psi, 0.0, NUM_PATTERNS* (ORDER - 1)*sizeof(double));
      fill_n(ones, NUM_PATTERNS, 1.0); // create ones vector

      if (form == 0) {
	    computeIdentityBasisFuncs( x, psi );	    
      }
      else {
	    computePolynomialBasisFuncs( x, psi );
      }
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

void computePseudoInverse( const double* Phi , double *phiPsuedoInverse ){
      double *A = (double *)mkl_malloc( ORDER*ORDER*sizeof( double ), 64 );
      double alpha = 1.0;
      double beta = 0.0;
      //declare MKL variables for inverse calculation
      int LWORK = ORDER*ORDER;
      int INFO;
      int *IPIV = (int *)mkl_malloc( (ORDER+1)*sizeof( int ), 64 );
      double *WORK = (double *)mkl_malloc( LWORK*sizeof( double ), 64 );

      memset( A, 0.0, ORDER* ORDER*sizeof(double));

      //calculate product = Phi' * Phi = A
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
		  ORDER, ORDER, NUM_PATTERNS, alpha, Phi,
		  ORDER, Phi, ORDER, beta, A, ORDER);

      //calculate inverse A = (A)^-1
      dgetrf( &ORDER, &ORDER, A, &ORDER, IPIV, &INFO );
      dgetri( &ORDER, A, &ORDER, IPIV, WORK, &LWORK, &INFO );

      // A * phi' = moore penrose!
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
		  ORDER, NUM_PATTERNS, ORDER, alpha, A,
		  ORDER, Phi, ORDER, beta, phiPsuedoInverse, NUM_PATTERNS);

      mkl_free( A );
      mkl_free( IPIV );
      mkl_free( WORK );
}

void solveNormalEquations( const double *inversePhi, const double *t, double *w ){
      // compute normal equations...
      //final solution should just be Moore-Penrose * t
      double alpha = 1.0;
      double beta = 0.0;
      cblas_dgemv( CblasRowMajor, CblasNoTrans, ORDER, NUM_PATTERNS,
      		   alpha, inversePhi, NUM_PATTERNS, t, 1, beta, w, 1);
}
void computeOutputs( const double *x, const double *w , double *y ){
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    double temp = 0.0;
	    for (int j = 0; j < ORDER; ++j) {
		  temp += w[j]*x[i];
	    }
	    y[i] = temp;
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
void setRandomWeights( double *weights ){
      for (int i = 0; i < ORDER; ++i) {
	    double temp = fRand( -10.0, 10.0);
	    weights[i] = temp;
      }
}

double fRand( const double fMin, const double fMax){
      double f = (double)rand() / RAND_MAX;
      return fMin + f * (fMax - fMin);
}
