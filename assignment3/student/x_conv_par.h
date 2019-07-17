#ifndef _X_CONV_PAR
#define _X_CONV_PAR
#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <png++/png.hpp>
#include <omp.h>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

Image x_convolution(Image &image, Matrix &filter, int num_thrds){
    assert(image.size()==3 && filter.size()!=0);

    int height = image[0].size();
    int width = image[0][0].size();
    int filterHeight = filter.size();
    int filterWidth = filter[0].size();
    int newImageHeight = height-filterHeight+1;
    int newImageWidth = width-filterWidth+1;
    int d,i,j,h,w;

    Image newImage(3, Matrix(newImageHeight, Array(newImageWidth)));

    for (d=0 ; d<3 ; d++) {
#pragma omp parallel for collapse(2) private(i,j,h,w) shared(newImage,image,filter)
        for (i=0 ; i<newImageHeight ; i++) {
            for (j=0 ; j<newImageWidth ; j++) {
                double p = newImage[d][i][j];
#pragma omp simd reduction(+:p)
                for (h=0 ; h<filterHeight ; h++) {
                    for (w=0 ; w<filterWidth ; w++) {
                        p += filter[h][w]*image[d][i+h][j+w];
                    }
                }
                newImage[d][i][j] = p;
            }
        }
    }

    return newImage;
}

#endif // !_X_CONV_PAR_
