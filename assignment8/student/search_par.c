#include <string.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include "search.h"
#include "helper.h"

void search_text (char* text, int num_lines, int line_length, char* search_string, int *occurences)
{
  /*
    Counts occurences of substring "search_string" in "text". "text" contains multiple lines and each line
    has been placed at text + line_length * num_lines since line length in the original text file can vary.
    "line_length" includes space for '\0'.

    Writes result at location pointed to by "occurences".


    *************************** PARALLEL VERSION **************************

    NOTE: For the parallel version, distribute the lines to each processor. You should only write
    to "occurences" from the root process and only access the text pointer from the root (all other processes
    call this function with text = NULL) 
  */

  // Write your parallel solution here
  int rank, num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  
  int num_lines_each = num_lines / num_procs,
      plus_one_count = num_lines % num_procs;

  int i, *displs, *scounts, offset;
    
  displs = (int *)malloc(num_procs*sizeof(int));
  scounts = (int *)malloc(num_procs*sizeof(int));
  offset = 0;
  for (i = 0; i < plus_one_count; i++) {
    displs[i] = offset;
    scounts[i]= (num_lines_each + 1) * line_length;
    offset += scounts[i];
  }
  for (i = plus_one_count; i < num_procs; i++) {
    displs[i] = offset;
    scounts[i] = num_lines_each * line_length;
    offset += scounts[i];
  }
  
  if (rank < plus_one_count) num_lines_each += 1;
  char * text_each = malloc(num_lines_each * line_length * sizeof(char));
  MPI_Scatterv(text, scounts, displs, MPI_CHAR, text_each, num_lines_each * line_length, MPI_CHAR, 0, MPI_COMM_WORLD);
  
  int running_count = 0;
  for (i = 0; i < num_lines_each; i++)
    {
      running_count += count_occurences(text_each + i * line_length, search_string);
    }
    
  MPI_Reduce(&running_count, occurences, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  
  free(displs);
  free(scounts);
  free(text_each);
}
