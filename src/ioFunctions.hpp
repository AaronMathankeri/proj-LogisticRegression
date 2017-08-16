/**
 *   \file ioFunctions.hpp
 *   \brief A Documented file.
 *
 *  Detailed description
 *
 */

#ifndef IOFUNCTIONS_H
#define IOFUNCTIONS_H

#include <fstream>
#include <sstream>
#include "parameters.hpp"

using namespace std;
void printVector( const double *x , const int length );
void printFeatures( const double *x1, const double *x2, const int length );
void printMatrix( const double *x, const int nRows, const int nCols);
void loadData( double *x , const string fileName );

#endif /* IOFUNCTIONS_H */
