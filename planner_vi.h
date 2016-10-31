//
// Created by Saipraveen B on 26/10/16.
//

#include<iostream>
#include<stdio.h>
#include <vector>

#define K 1
#define A 4
#define ITERS 40
#define GAMMA 0.90

// TODO: Optimize.
std::vector<std::vector<double>> _Iterate( std::vector<std::vector<double> > R, std::vector<std::vector<double> > V, std::vector< std::vector< std::vector<double> > > P ){
  std::vector<std::vector<double>> V_next( V.size(), std::vector<double>(V[0].size(), -100.0) );

  int h = R.size();
  int w = R[0].size();
  // Roughly 28. + 2 padding.
  for( int ty = 1; ty < h+1; ty++){
    // Roughly 28. + 2 padding.
    for( int tx = 1; tx < w+1; tx++ ){
      // Roughly 4-5.
      double max = -200;
      for( int a = 0; a < A; a++ ){
        double _v = 0;
        // Roughly 3
        for( int fy = -K; fy <= +K; fy++ ){
          // Roughly 3
          for( int fx = -K; fx <= +K; fx++ ){

            // Possible exploitation of cache locality here.
            // Could potentially reorder this.
            //if( tx == 10 && ty == 9 && a == 3 && fx == 0 && fy == 1){
            //  int p = 0;
            //}

            _v += V[ty + fy][tx + fx] * P[a][(ty-1) * w + (tx-1)][(fy+K) * (2 * K + 1 ) + (fx+K)];
          }
        }

        if( _v > max ) max = _v;
      }
      // Put it into the output matrix.
      V_next[ty][tx] = (R[ty-1][tx-1] - 0.1) + GAMMA * max;
      int u = 0;
    }
  }
  return V_next;
}

// Uses Value Iteration to compute the value functions for each square.
std::vector<std::vector<double>> ValueIterate(std::vector<std::vector<double> > R, std::vector< std::vector< std::vector<double> > > P ){
  std::vector<std::vector<double>> V( R.size() + 2, std::vector<double>(R[0].size() + 2, -100.0) );
  for( int x = 1; x <  V.size()-1; x++ ){
    for( int y = 1; y <  V[0].size()-1; y++ ) {
      V[x][y] = R[x-1][y-1];
    }
  }

  for( int i = 0; i < ITERS; i++ ){
    V = _Iterate(R,V,P);
  }

  return V;
}
