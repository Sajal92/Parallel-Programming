#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <pthread.h>

#include "mandelbrot_set.h"

#define CHUNK_LEN 16

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int * g_y_index;
int g_x_resolution, g_y_resolution;
int g_max_iter;
double g_view_x0, g_view_y1;
double g_x_stepsize, g_y_stepsize;
int g_palette_shift;
unsigned char ***g_image;

void* work(void* args) {
	int y_start, y_end;
	
    double y;
	double x;

	complex double Z;
	complex double C;

	int k;

    for (;;) {
        pthread_mutex_lock(&mutex);
        if (g_y_resolution - *g_y_index < 1) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        
        y_start = *g_y_index;
        *g_y_index += CHUNK_LEN;
        pthread_mutex_unlock(&mutex);
        
        y_end = g_y_resolution > y_start + CHUNK_LEN ? y_start + CHUNK_LEN : g_y_resolution;
        for (int i = y_start; i < y_end; i++)
        {
            for (int j = 0; j < g_x_resolution; j++)
            {
                y = g_view_y1 - i * g_y_stepsize;
                x = g_view_x0 + j * g_x_stepsize;

                Z = 0 + 0 * I;
                C = x + y * I;

                k = 0;

                do
                {
                    Z = Z * Z + C;
                    k++;
                } while (cabs(Z) < 2 && k < g_max_iter);

                if (k == g_max_iter)
                {
                    memcpy(((unsigned char(*)[g_x_resolution][3])g_image)[i][j], "\0\0\0", 3);
                }
                else
                {
                    int index = (k + g_palette_shift)
                                % (sizeof(colors) / sizeof(colors[0]));
                    memcpy(((unsigned char(*)[g_x_resolution][3])g_image)[i][j], colors[index], 3);
                }
            }
        }
    }

	return NULL;
}

void mandelbrot_draw(int x_resolution, int y_resolution, int max_iter,
			                double view_x0, double view_x1, double view_y0, double view_y1,
						                double x_stepsize, double y_stepsize,
									                int palette_shift, unsigned char (*image)[x_resolution][3],
																	 int num_threads) {
	pthread_t threads[num_threads];
	int y_index = 0;
    g_y_index = &y_index;
    g_x_resolution = x_resolution;
    g_y_resolution = y_resolution;
	g_max_iter = max_iter;
    g_view_x0 = view_x0;
    g_view_y1 = view_y1;
    g_x_stepsize = x_stepsize;
    g_y_stepsize = y_stepsize;
    g_palette_shift = palette_shift;
    g_image = (unsigned char***)image;
    
    for (int t = 0; t < num_threads; t++) {
		pthread_create(&threads[t], NULL, work, NULL);
	}

	for (int t = 0; t < num_threads; t++) {
		pthread_join(threads[t], NULL);
	}
}
