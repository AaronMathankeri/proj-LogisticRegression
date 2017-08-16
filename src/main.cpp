//Copyright (C) 2017 Aaron Mathankeri (aaron@quantifiedag.com)
//License: QuantifiedAG Software License.  See LICENSE.txt for full license.

/*
 *   \file example.cpp
 *   \brief A Documented file.
 *
 *  Detailed description
 *
 */
#include <iostream>
#include "mkl.h"
#include "ioFunctions.hpp"
#include "parameters.hpp"
#include "linearParameterModels.hpp"

using namespace std;

int main(int argc, char *argv[])
{
      cout << " Aaron's Back." << endl;

      //--------------------------------------------------------------------------------
      // declare variables for calculations
      double *x1, *x2, *t;

      x1 = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      x2 = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      t = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      
      memset( x1, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( x2, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( t, 0.0,  NUM_PATTERNS * sizeof(double));
      //--------------------------------------------------------------------------------
      //read data
      string x1File = "./data/train/x1.txt";
      string x2File = "./data/train/x2.txt";
      string targetsFile = "./data/train/class.txt";

      loadData( x1 , x1File );
      loadData( x2 , x2File );
      loadData( t , targetsFile );
      
      cout << "Features" << endl;
      cout << "x1\t | \tx2" << endl;
      printFeatures( x1, x2, NUM_PATTERNS );
      
      cout << "Targets" << endl;
      printVector( t, NUM_PATTERNS );
      //--------------------------------------------------------------------------------
      printf ("\n Deallocating memory \n\n");
      mkl_free( x1 );
      mkl_free( x2 );
      mkl_free( t );
      printf (" Example completed. \n\n");

      return 0;
}
