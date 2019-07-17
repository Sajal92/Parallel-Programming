#include <iostream>
#include <png++/png.hpp>
#include <assert.h>
#include"x_conv.h"

using namespace std;
int main() {



    Matrix filter = getGaussian(13, 13, 1000);
    Image image = loadImage("TUM.png");

    Image newImage_ref;
    Image newImage;

    int height_ref,width_ref;
    int height_par,width_par;

    for (int t = 2; t < 33; t+=30)
    {
        newImage_ref = applyFilter_ref(&image, filter, t);
        newImage     = applyFilter(&image, filter, t);
         
        height_par = newImage[0].size();   
        width_par = newImage[0][0].size();
        height_ref = newImage_ref[0].size();
        width_ref = newImage_ref[0][0].size();
        assert(height_par == height_ref);
        assert(width_par == width_ref);

        //Compare images by each pixel

	for (int i=0 ; i<height_ref ; i++) {
            for (int j=0 ; j<width_ref ; j++) {
		for (int d=0 ; d<3 ; d++) {
		    if(newImage_ref[d][i][j] != newImage[d][i][j])
		    {
	   	       fprintf(stderr, "Computation with %d threads failed\n", t);
                       fprintf(stderr, "Image differs from the one produced by the sequential version\n");
                       return 1;
		    }
		}
	    }
	}
    }
    return 0;
}
