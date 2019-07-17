#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<string.h>
#include"helper.h"
#include"life.h"
#include"gui.h"

#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3
#define UL    4
#define DR    5
#define UR    6
#define DL    7

void evolve_(int height, int width, int grid[height][width])
{
    int temp[height][width], i, j;

    for (i = 1; i < height-1; i++) {
        if (i >= 3)
            memcpy(&grid[i-2][1], &temp[i-2][1], (width-2)*sizeof(int));
        for (j = 1; j < width-1; j++) {
            int sum = grid[i-1][j-1] + grid[i-1][j] + grid[i-1][j+1]
                    + grid[i][j-1] + grid[i][j+1]
                    + grid[i+1][j-1] + grid[i+1][j] + grid[i+1][j+1];
            if (grid[i][j] == 0) {
                // reproduction
                if (sum == 3)
                    temp[i][j] = 1;
                else
                    temp[i][j] = 0;
            } else { // alive
                // stays alive
                if (sum == 2 || sum == 3)
                    temp[i][j] = 1;
                // dies due to under or overpopulation
                else
                    temp[i][j] = 0;
            }
        }
        if (i == height-2) {
            memcpy(&grid[i-1][1], &temp[i-1][1], (width-2)*sizeof(int));
            memcpy(&grid[i][1], &temp[i][1], (width-2)*sizeof(int));
        }
    }
}

void simulate(int height, int width, int grid[height][width], int num_iterations)
{
  /*
    Write your parallel solution here. You first need to distribute the data to all of the
    processes from the root. Note that you cannot naively use the evolve function used in the
    sequential version of the code - you might need to rewrite it depending on how you parallelize
    your code.

    For more details, see the attached readme file.
  */
    int rank, num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    MPI_Comm cartcomm;
    int nbrs[8], dims[2]={0,0}, periods[2]={1,1}, reorder=0, coords[2];
    
    MPI_Dims_create(num_procs, 2, dims);
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartcomm);
    MPI_Comm_rank(cartcomm, &rank);
    MPI_Cart_coords(cartcomm, rank, 2, coords);
    MPI_Cart_shift(cartcomm, 0, 1, &nbrs[UP], &nbrs[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &nbrs[LEFT], &nbrs[RIGHT]);
    // Get neighbors at 4 corners
    if (rank % dims[1] != 0) {
        nbrs[UL] = nbrs[UP]-1;
        nbrs[DL] = nbrs[DOWN]-1;
    } else {
        nbrs[UL] = nbrs[UP]+(dims[1]-1);
        nbrs[DL] = nbrs[DOWN]+(dims[1]-1);
    }
    if (rank % dims[1] != dims[1]-1) {
        nbrs[DR] = nbrs[DOWN]+1;
        nbrs[UR] = nbrs[UP]+1;
    } else {
        nbrs[DR] = nbrs[DOWN]-(dims[1]-1);
        nbrs[UR] = nbrs[UP]-(dims[1]-1);
    }
//     printf("rank %d, dims %d %d, coords %d %d, nbrs %d %d %d %d %d %d %d %d\n", rank, dims[0], dims[1], coords[0], coords[1], nbrs[0], nbrs[1], nbrs[2], nbrs[3], nbrs[4], nbrs[5], nbrs[6], nbrs[7]);
    
    // calculate sizes
    int block_height = ((height-2)-1) / dims[0] + 1, block_width = ((width-2)-1) / dims[1] + 1,
        padded_height = block_height * dims[0], padded_width = block_width * dims[1],
        local_height = padded_height == height-2 || rank / dims[1] != dims[0]-1 ? block_height : block_height - (height-2) % dims[0],
        local_width = padded_width == width-2 || rank % dims[1] != dims[1]-1 ? block_width : block_width - (width-2) % dims[1],
        i, j;
    // block array
    int block_array[block_height][block_width];
//     printf("rank %d: block height %d width %d, padded height %d width %d, local height %d width %d\n",
//             rank, block_height, block_width, padded_height, padded_width, local_height, local_width);
    
    // copy the original grid to a new one, pad it if necessary
//     int padded_grid[padded_height][padded_width];
//     if (rank == 0) {
//         memset(padded_grid, 0, padded_height * padded_width * sizeof(int));
//         print_grid("padded set:", padded_height, padded_width, padded_grid);
//         for (i = 0; i < height-2; i++)
//             memcpy(&padded_grid[i][0], &grid[i+1][1], (width-2)*sizeof(int));
//         print_grid("padded pad:", padded_height, padded_width, padded_grid);
//     }
    
    // create a new type of submatrix
    MPI_Datatype blocktype, blocktype2;
    MPI_Type_vector(block_height, block_width, width, MPI_INT, &blocktype2);
    MPI_Type_create_resized(blocktype2, 0, sizeof(int), &blocktype);
    MPI_Type_commit(&blocktype);
    
    // create parameters for scatterv calls
    int count_at_root[num_procs], displs[num_procs];
    for (i = 0; i < dims[0]; i++) {
        for (j = 0; j < dims[1]; j++) {
            count_at_root[i*dims[1]+j] = 1;
            displs[i*dims[1]+j] = i*block_height*width + j*block_width;
        }
    }
    
//     if (rank == 0)
//         print_grid("grid:", height, width, grid);
    
    MPI_Scatterv(&grid[1][1], count_at_root, displs, blocktype, block_array, block_height*block_width, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&blocktype);
//     char buffer[50];
//     sprintf(buffer, "rank %d:", rank);
//     print_grid(buffer, block_height, block_width, block_array);
    // local array
    int local_array[local_height+2][local_width+2];
    for (i = 1; i < local_height+1; i++)
        memcpy(&local_array[i][1], &block_array[i-1][0], local_width*sizeof(int));
    
    // local memory from here has its part of the grid
    //int inbuf_h[2][local_width], inbuf_v[2][local_height], inbuf_corners[4];
    int tag = 13, tag_U = tag, tag_D = tag, tag_L = tag, tag_R = tag, tag_UL = tag, tag_DR = tag, tag_UR = tag, tag_DL = tag;
    if (dims[0] <= 2) {
        tag_U += UP;
        tag_D += DOWN;
    }
    if (dims[1] <= 2) {
        tag_L += LEFT;
        tag_R += RIGHT;
    }
    if (dims[0] <= 2 && dims[1] <= 2) {
        tag_UL += UL;
        tag_DR += DR;
        tag_UR += UR;
        tag_DL += DL;
    }
    MPI_Request reqs[16];
    MPI_Status stats[16];
    
    MPI_Datatype column_vector;
    MPI_Type_vector(local_height, 1, local_width+2, MPI_INT, &column_vector);
    MPI_Type_commit(&column_vector);
    for (int it = 0; it < num_iterations; it++) {
        MPI_Isend(&local_array[1][1], local_width, MPI_INT, nbrs[UP], tag_U, MPI_COMM_WORLD, &reqs[UP]);
        MPI_Irecv(&local_array[0][1], local_width, MPI_INT, nbrs[UP], tag_D, MPI_COMM_WORLD, &reqs[UP+8]);
        MPI_Isend(&local_array[local_height][1], local_width, MPI_INT, nbrs[DOWN], tag_D, MPI_COMM_WORLD, &reqs[DOWN]);
        MPI_Irecv(&local_array[local_height+1][1], local_width, MPI_INT, nbrs[DOWN], tag_U, MPI_COMM_WORLD, &reqs[DOWN+8]);
        
        MPI_Isend(&local_array[1][1], 1, column_vector, nbrs[LEFT], tag_L, MPI_COMM_WORLD, &reqs[LEFT]);
        MPI_Irecv(&local_array[1][0], 1, column_vector, nbrs[LEFT], tag_R, MPI_COMM_WORLD, &reqs[LEFT+8]);
        MPI_Isend(&local_array[1][local_width], 1, column_vector, nbrs[RIGHT], tag_R, MPI_COMM_WORLD, &reqs[RIGHT]);
        MPI_Irecv(&local_array[1][local_width+1], 1, column_vector, nbrs[RIGHT], tag_L, MPI_COMM_WORLD, &reqs[RIGHT+8]);
        
        MPI_Isend(&local_array[1][1], 1, MPI_INT, nbrs[UL], tag_UL, MPI_COMM_WORLD, &reqs[UL]);
        MPI_Irecv(&local_array[0][0], 1, MPI_INT, nbrs[UL], tag_DR, MPI_COMM_WORLD, &reqs[UL+8]);
        MPI_Isend(&local_array[local_height][local_width], 1, MPI_INT, nbrs[DR], tag_DR, MPI_COMM_WORLD, &reqs[DR]);
        MPI_Irecv(&local_array[local_height+1][local_width+1], 1, MPI_INT, nbrs[DR], tag_UL, MPI_COMM_WORLD, &reqs[DR+8]);
        MPI_Isend(&local_array[1][local_width], 1, MPI_INT, nbrs[UR], tag_UR, MPI_COMM_WORLD, &reqs[UR]);
        MPI_Irecv(&local_array[0][local_width+1], 1, MPI_INT, nbrs[UR], tag_DL, MPI_COMM_WORLD, &reqs[UR+8]);
        MPI_Isend(&local_array[local_height][1], 1, MPI_INT, nbrs[DL], tag_DL, MPI_COMM_WORLD, &reqs[DL]);
        MPI_Irecv(&local_array[local_height+1][0], 1, MPI_INT, nbrs[DL], tag_UR, MPI_COMM_WORLD, &reqs[DL+8]);
        
        MPI_Waitall(16, reqs, stats);
        
//         memcpy(&local_array[0][1], inbuf_h[0], local_width*sizeof(int));
//         memcpy(&local_array[local_height+1][1], inbuf_h[1], local_width*sizeof(int));
//         for (i = 1; i < local_height+1; i++)
//             local_array[i][0] = inbuf_v[0][i-1];
//         for (i = 1; i < local_height+1; i++)
//             local_array[i][local_width+1] = inbuf_v[1][i-1];
//         local_array[0][0] = inbuf_corners[UL-4];
//         local_array[local_height+1][local_width+1] = inbuf_corners[DR-4];
//         local_array[0][local_width+1] = inbuf_corners[UR-4];
//         local_array[local_height+1][0] = inbuf_corners[DL-4];
        
//         sprintf(buffer, "rank %d local array iteration %d:", rank, it);
//         print_grid(buffer, local_height+2, local_width+2, local_array);
        
        evolve_(local_height+2, local_width+2, local_array);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Type_free(&column_vector);
    
    MPI_Datatype blocktype_, blocktype_2;
    MPI_Type_vector(block_height, block_width, padded_width, MPI_INT, &blocktype_2);
    MPI_Type_create_resized(blocktype_2, 0, sizeof(int), &blocktype_);
    MPI_Type_commit(&blocktype_);
    
    int padded_grid[padded_height][padded_width];
    // block array
    for (i = 1; i < local_height+1; i++)
        memcpy(&block_array[i-1][0], &local_array[i][1], local_width*sizeof(int));
    MPI_Gatherv(block_array, block_height*block_width, MPI_INT, padded_grid, count_at_root, displs, blocktype_, 0, MPI_COMM_WORLD);
    MPI_Type_free(&blocktype_);
    
    // copy data to the original grid
    if (rank == 0) {
        for (i = 0; i < height-2; i++)
            memcpy(&grid[i+1][1], &padded_grid[i][0], (width-2)*sizeof(int));
    }
}
