#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <png++/png.hpp>
#include "x_conv.h"
#include <chrono>
#include <unistd.h>

using namespace std;

int main(int argc, char **argv)
{
    int num_threads = 1;
    char file_name[256] = "TUM.png";
    int no_output = 0;
    int c;
    while ((c = getopt(argc, argv, "t:n:f:")) != -1)
    {
        switch (c)
        {
           case 't':
                 if (sscanf(optarg, "%d", &num_threads) != 1)
                    goto error;
                 break;
           case 'n':
                 if (sscanf(optarg, "%d", &no_output) != 1)
                    goto error;
                 break;
           case 'f':
                 strncpy(file_name, optarg, sizeof(file_name));
                 break;
           case '?':
           error: printf(
                            "Usage:\n"
                            "-t \t number of threads used in computation(default: 1)\n"
                            "-f \t output file name(default: TUM.png)\n"
                                "-n \t no output(default: 0)\n"
                            "\n"
                            "Example:\n"
                            "%s -t 4 -f TUM.png\n",
                            argv[0]);
                  exit(EXIT_FAILURE);
                  break;
    	}
    }


    Matrix filter = getGaussian(11, 11, 1000);

    //cout << "Loading image..." << endl;
    Image image = loadImage(file_name);
    //cout << "Applying filter..." << endl;

    auto start = std::chrono::high_resolution_clock::now();
    Image newImage = applyFilter(&image, filter, num_threads);
    auto end = std::chrono::high_resolution_clock::now();

    //cout << "Saving image..." << endl;
    saveImage(newImage, "newImage.png");
    //cout << "Done!" << endl;
    
    std::chrono::duration<double> diff = end-start;
    cout << "Time: " << diff.count() << " seconds" << endl;
}
