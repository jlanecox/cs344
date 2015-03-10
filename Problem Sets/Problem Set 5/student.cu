/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include <thrust/sort.h>

#include "utils.h"
#include "reference_calc.h"

#define SHARED_VER 1
#define GLOBAL_VER 0

#if GLOBAL_VER
__global__
void yourHisto(const unsigned int* const dvals, //INPUT
               unsigned int* const dhisto,      //OUPUT
               const int numVals)
{
    const unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if( pos >= numVals )
        return;

    // global version
    unsigned int bin = dvals[pos];
    atomicAdd( &(dhisto[bin]), 1 );
}
#endif

#if SHARED_VER
__global__
void yourHisto(const unsigned int* const dvals, //INPUT
               unsigned int* const dhisto,      //OUPUT
               const int numVals, const unsigned int numBins,
               const unsigned int valsPerThread)
{
    extern __shared__ unsigned int sdata[];

    // numberThreadsPerBlock = blockDim.x
    // valsPerThread is given
    // startPosition is blockIdx.x*blockDim.x*valsPerThread + threadIdx.x*valsPerThread
    const unsigned int startpos = valsPerThread*(blockIdx.x*blockDim.x + threadIdx.x);
    const unsigned int tid = threadIdx.x;

    if( startpos+valsPerThread-1 >= numVals )
        return;

    const unsigned int numBinsPerThread = numBins / blockDim.x;

    //zero out sdata
    for(unsigned int i=0; i < numBinsPerThread; ++i)
    {
      sdata[numBinsPerThread*tid+i] = 0;
    }
    __syncthreads();

    for( unsigned int i=0; i < valsPerThread; ++i )
    {
      unsigned int bin = dvals[startpos+i];
      atomicAdd( &(sdata[bin]), 1 );
    }
    __syncthreads();

    for(unsigned int i=0; i < numBinsPerThread; ++i)
    {
      atomicAdd( &(dhisto[numBinsPerThread*tid +i]), sdata[numBinsPerThread*tid +i] );
    }
}
#endif

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  // d_histo is already cleared

  //if you want to use/launch more than one kernel,
  //feel free

#if GLOBAL_VER
  // Launch the yourHisto kernel
  const dim3 bsHisto( 512, 1, 1 );
  const dim3 gsHisto( (numElems + bsHisto.x -1) / bsHisto.x, 1, 1 );
  yourHisto<<< gsHisto, bsHisto >>>( d_vals, d_histo, numElems );
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
#endif

#if SHARED_VER
  // Launch the yourHisto kernel
  const unsigned int numValsPerThread = 4;
  const unsigned int numThreadsPerBlk = 512;

  const dim3 bsHisto( numThreadsPerBlk, 1, 1 );
  const dim3 gsHisto( (numElems + bsHisto.x + numValsPerThread -1) / (bsHisto.x*numValsPerThread), 1, 1 );

  yourHisto<<< gsHisto, bsHisto, sizeof(unsigned int)*numBins >>>( d_vals, d_histo,
                                                                  numElems, numBins,
                                                                  numValsPerThread );
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
#endif



 /* delete[] h_vals;
  delete[] h_histo;
  delete[] your_histo;*/
}
