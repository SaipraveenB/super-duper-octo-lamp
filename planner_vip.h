//
// Created by Saipraveen B on 31/10/16.
//

#ifndef MONTECARLOSOLVER_PLANNER_VIP_H_H
#define MONTECARLOSOLVER_PLANNER_VIP_H_H

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
std::vector< std::vector<std::vector<double>> > _Iterate_pR( std::vector<std::vector<double> > R, std::vector< std::vector<std::vector<double> > > V, std::vector< std::vector< std::vector<double> > > P, std::vector< std::vector<double> > pR){
  std::vector<std::vector<std::vector<double>>> V_next( V.size(), std::vector< std::vector<double> >( V[0].size(), std::vector<double>( 2, -100.0) ) );

  int h = R.size();
  int w = R[0].size();
  // Roughly 28. + 2 padding.
  for( int ty = 1; ty < h+1; ty++){
    // Roughly 28. + 2 padding.
    for( int tx = 1; tx < w+1; tx++ ){
      // Roughly 4-5.
      for( int startp = 0; startp < 2; startp ++ ) {
        double max = -200;
        for (int a = 0; a < A * 2; a++) {
          double _v = 0;
          // Roughly 3
          for (int fy = -K; fy <= +K; fy++) {
            // Roughly 3
            for (int fx = -K; fx <= +K; fx++) {

              for (int endp = 0; endp < 2; endp++) {
                // Possible exploitation of cache locality here.
                // Could potentially reorder this.
                //if( tx == 10 && ty == 9 && a == 3 && fx == 0 && fy == 1){
                //  int p = 0;
                //}
                if (endp == 1 && startp == 0 && a >= A)
                  _v += (pR[ty-1][tx-1] + V[ty + fy][tx + fx][endp])
                      * P[a % A][(ty - 1) * w + (tx - 1)][(fy + K) * (2 * K + 1) + (fx + K)];
                else if (endp == 0 && startp == 0 && a < A)
                  _v += V[ty + fy][tx + fx][endp] * P[a % A][(ty - 1) * w + (tx - 1)][(fy + K) * (2 * K + 1) + (fx + K)];
                //else if (endp == 1 && startp == 1 && a < A)
                //  _v += V[ty + fy][tx + fx][endp] * P[a % A][(ty - 1) * w + (tx - 1)][(fy + K) * (2 * K + 1) + (fx + K)];

              }
            }
          }

          if (_v > max) max = _v;

        }

        // Put it into the output matrix.
        V_next[ty][tx][startp] = (R[ty - 1][tx - 1]) + GAMMA * max;
        int u = 0;

      }
    }
  }
  return V_next;
}

// Uses Value Iteration to compute the value functions for each square.
std::vector<std::vector<double>> ValueIterate_pR(std::vector<std::vector<double> > R, std::vector< std::vector< std::vector<double> > > P, std::vector<std::vector<double> > pR ){
  std::vector<std::vector<std::vector<double>>> V( R.size() + 2, std::vector< std::vector<double> >( R[0].size() + 2, std::vector<double>( 2, -100.0) ) );
  for( int x = 1; x <  V.size()-1; x++ ){
    for( int y = 1; y <  V[0].size()-1; y++ ) {
      V[x][y][0] = R[x-1][y-1];
    }
  }

  for( int i = 0; i < ITERS; i++ ){
    V = _Iterate_pR(R,V,P,pR);
  }

  std::vector< std::vector< double > > V_slice( V.size(), std::vector<double>( V[0].size(), 0) );

  for( int i = 0; i < V.size(); i++ ){
    for( int j = 0; j < V[0].size(); j++ ){
      V_slice[i][j] = V[i][j][0];
    }
  }

  return V_slice;
}

#endif //MONTECARLOSOLVER_PLANNER_VIP_H_H
