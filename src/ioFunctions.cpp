#include "ioFunctions.hpp"

void printVector( const double *x , const int length ){
      for (int i = 0; i < length; i++) {
	    printf ("%12.5f", x[i]);
	    printf ("\n");
      }
}

void printFeatures( const double *x1, const double *x2, const int length ){
      for (int i = 0; i < length; i++) {
	    printf ("%5.3f", x1[i]);
	    printf ("\t\t%5.3f", x2[i]);
	    printf ("\n");
      }
}

void printMatrix( const double *x, const int nRows, const int nCols){
      for (int i=0; i < nRows; i++) {
	    for (int j=0; j < nCols; j++) {
		  printf ("%12.3f", x[i*nCols +j]);
	    }
	    printf ("\n");
      }
}

void loadData( double *x , const string fileName ){
      ifstream file  ( fileName );
      if(file.is_open()) {
	    for (int i = 0; i < NUM_PATTERNS; ++i) {
		  file >> x[i];
	    }
      }
}
