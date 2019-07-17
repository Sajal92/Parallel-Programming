#include "dgemm.h"
#include <immintrin.h>
#include <inttypes.h>

// from https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
float sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}


void dgemm(float *a, float *b, float *c, int n) {
    int32_t mask_src[] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};
    
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            for(int k = 0; k < n; k+=8){
                //c[i * n + j] += a[i * n  + k] * b[j * n  + k];
                int r = n - k;
                __m256 a_k, b_k;
                if (r < 8) {
                    __m256i imask = _mm256_loadu_si256((__m256i const *)(mask_src + 8 - r));
                    a_k = _mm256_maskload_ps(a+i*n+k, imask);
                    b_k = _mm256_maskload_ps(b+j*n+k, imask);
                } else {
                    a_k = _mm256_loadu_ps(a+i*n+k);
                    b_k = _mm256_loadu_ps(b+j*n+k);
                }
                __m256 mul = _mm256_mul_ps(a_k, b_k);
                
                c[i*n+j] += sum8(mul);
            }
        }
    }
}

