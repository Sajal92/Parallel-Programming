#include "familytree.h"
#include <omp.h>

int par_traverse(tree *node) {
    if (node == NULL) return 0;
    
    int father_iq, mother_iq;
    
#pragma omp task shared(father_iq) 
    father_iq = par_traverse(node->father);
//#pragma omp task shared(mother_iq) 
    mother_iq = par_traverse(node->mother);
    
#pragma omp taskwait
    node->IQ = compute_IQ(node->data, father_iq, mother_iq);
    genius[node->id] = node->IQ;

    return node->IQ;
}

int traverse(tree *node, int numThreads){
    omp_set_num_threads(numThreads);
#pragma omp parallel 
    {
#pragma omp single
        par_traverse(node);
    }
    
    return node->IQ;
}
