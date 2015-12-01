/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "reference_calc.h"
#include "utils.h"

  // min = -3.10921
  // max = 2.26509
  // range = 5.37429
__global__ void reduce_min_and_max_1D(float* d_minOut, float* d_maxOut,
                                      const float* const d_inMin, const float* const d_inMax,
                                     const int numRows, const int numCols, const int flag)
{
  extern __shared__ float sdata[];

  const int myId = ( blockIdx.x * (blockDim.x*2) ) + threadIdx.x;
  //const int myId = ( blockIdx.x * (blockDim.x) ) + threadIdx.x;
  const int tid = threadIdx.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if( myId >= (numCols*numRows ) )
    return;

  float* smaxdata = &sdata[0];
  float* smindata = &sdata[blockDim.x];

  const float* inMin = d_inMin;
  const float* inMax = d_inMax;

  // only use d_Min
  if( flag == 1 )
  {
    inMax = d_inMin;
  }
  // use only d_Max
  else if( flag == 2 )
  {
    inMin = d_inMax;
  }

  smindata[tid] = min( inMin[myId], inMin[myId + blockDim.x]);
  smaxdata[tid] = max( inMax[myId], inMax[myId + blockDim.x]);
  //smindata[tid] = inMin[myId];
  //smaxdata[tid] = inMax[myId];

  __syncthreads();

  for( unsigned int s=(blockDim.x/2); s>0; s>>=1 )
  {
    if( tid < s )
    {
      smindata[tid] = min( smindata[tid], smindata[tid + s] );
      smaxdata[tid] = max( smaxdata[tid], smaxdata[tid + s] );
    }
    __syncthreads();
  }

  /*
  if( tid < 32 )
  {
    smindata[tid] = min( smindata[tid], smindata[tid + 6] );
    smaxdata[tid] = max( smaxdata[tid], smaxdata[tid + 6] );

    smindata[tid] = min( smindata[tid], smindata[tid + 5] );
    smaxdata[tid] = max( smaxdata[tid], smaxdata[tid + 5] );

    smindata[tid] = min( smindata[tid], smindata[tid + 4] );
    smaxdata[tid] = max( smaxdata[tid], smaxdata[tid + 4] );

    smindata[tid] = min( smindata[tid], smindata[tid + 3] );
    smaxdata[tid] = max( smaxdata[tid], smaxdata[tid + 3] );

    smindata[tid] = min( smindata[tid], smindata[tid + 2] );
    smaxdata[tid] = max( smaxdata[tid], smaxdata[tid + 2] );

    smindata[tid] = min( smindata[tid], smindata[tid + 1] );
    smaxdata[tid] = max( smaxdata[tid], smaxdata[tid + 1] );
  }
  */

  if( tid == 0 )
  {
    d_minOut[blockIdx.x] = smindata[tid];
    d_maxOut[blockIdx.x] = smaxdata[tid];
  }
}




__global__ void gen_histo(unsigned int* d_out, const float* const d_in,
                          const float minLum, const float lumRange,
                          const int numBins, const int numRows, const int numCols)
{
//  extern __shared__ float sdata[];

  //const float lumRange = maxLum - minLum;

  const int myPos = (blockDim.x * blockIdx.x) + threadIdx.x;
  //const int tid = threadIdx.x;

  //const int myBlk1D_pos = threadIdx.x + blockDim.x * threadIdx.y;
  //const int totalBlockSamps = blockDim.x * blockDim.y;


  // function: bin = (lum[i] - lumMin) / lumRange * numBins
  const float item = d_in[myPos];
  const unsigned int bin = min( (unsigned int)(( item - minLum) / lumRange * (float)numBins), (unsigned int)(numBins-1) );
  //atomicAdd( &(sdata[bin]), 1);
  atomicAdd( &(d_out[bin]), 1);
}

__global__ void exclusive_scan(unsigned int* d_out, const unsigned int* const d_in,
                              const int numBins)
{
  // blelloch scan (reduce and downsweep)
  extern __shared__ unsigned int sidata[];

  const int tid = threadIdx.x;
  //const int myPos = threadIdx.x;// + (blockIdx.x * blockDim.x*2);
  int offset = 1;

/*
  // this should not happen
  if( myPos > numBins )
  {
    return;
  }
*/

  // copy data into shared mem
  sidata[2*tid] = d_in[2*tid+1];
  sidata[2*tid+1] = d_in[2*tid+1];
  __syncthreads();

  // reductions steps
  for(unsigned int s=(numBins>>1); s > 0; s >>= 1)
  {
    if(tid < s)
    {
      int a = offset*(2*tid+1)-1;
      int b = offset*(2*tid+2)-1;

      sidata[b] += sidata[a];
    }
    __syncthreads();

    offset *= 2;
  }

  // reduction step done, clear the last element
    if( tid == 0 ) { sidata[numBins-1] = 0; }

  // downsweep steps
  for(unsigned int s=1; s < numBins; s *= 2)
  {
    offset >>= 1;
    if( tid < s )
    {
      int a = offset*(2*tid+1)-1;
      int b = offset*(2*tid+2)-1;

      unsigned int tmp = sidata[a];
      sidata[a] = sidata[b];
      sidata[b] += tmp;
    }
    __syncthreads();
  }

  // copy to output
  d_out[2*tid]   = sidata[2*tid];
  d_out[2*tid+1] = sidata[2*tid+1];
}


#define MAX_THREADS 1024

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  const size_t numPixels = numRows * numCols;
  // const size_t cdf_len = numBins;
  // const size_t logLuminance_len = numPixels;

  const dim3 blockSize(numCols, 1, 1);
  const dim3 gridSize((numPixels + blockSize.x - 1) / blockSize.x, 1, 1);
  //TODO

  /*Here are the steps you need to implement*/
  // 1) find the minimum and maximum value in the input logLuminance channel
  //    store in min_logLum and max_logLum
  // myTODO: imp reduce with min

  // need space for each reduction of each block
  float *d_minOut = NULL;
  float *d_maxOut = NULL;

  const unsigned int numBlocks = gridSize.x * gridSize.y;

  checkCudaErrors( cudaMalloc(&d_minOut, numPixels*sizeof(float)) );
  checkCudaErrors( cudaMalloc(&d_maxOut, numPixels*sizeof(float)) );

  checkCudaErrors(cudaMemset(d_minOut, 0, sizeof(float)*numPixels));
  checkCudaErrors(cudaMemset(d_maxOut, 0, sizeof(float)*numPixels));

  const int numThreadsPerBlock = blockSize.x * blockSize.y;

  reduce_min_and_max_1D<<<gridSize, blockSize, 2*numThreadsPerBlock*sizeof(float)>>>
      ( d_minOut, d_maxOut, d_logLuminance, d_logLuminance, numRows, numCols, 1 );

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


  float* h_min = new float[numBlocks];
  float* h_max = new float[numBlocks];

  checkCudaErrors(cudaMemcpy(h_min, d_minOut, (numBlocks*sizeof(float)), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_max, d_maxOut, (numBlocks*sizeof(float)), cudaMemcpyDeviceToHost));

  min_logLum = 0.0f;
  max_logLum = 0.0f;
  for(unsigned int i=0; i < numBlocks; ++i)
  {
    min_logLum = min( min_logLum, h_min[i] );
    max_logLum = max( max_logLum, h_max[i] );
  }

  delete [] h_min; h_min = NULL;
  delete [] h_max; h_max = NULL;


  /////////////////////////////////////////
  // 2) subtract them to find the range  //
  float range = max_logLum - min_logLum;

  /*
  std::cout << "Min, Max, Range: "
            << min_logLum << " "
            << max_logLum << " "
            << range
            << "\nnumRows, numCols, numBins: "
            << numRows << " " << numCols << " " << numBins
            << std::endl;

  std::cout << "NumBlocks (gridX, gridY): " << numBlocks
            << " (" << gridSize.x << "," << gridSize.y
            << "), (blockX, blockY): " << blockSize.x
            << "," << blockSize.y << ")"
            << std::endl;
   */
  checkCudaErrors(cudaFree(d_minOut));
  checkCudaErrors(cudaFree(d_maxOut));



  // 3) generate a histogram of all the values in the logLuminance channel using
  //    the formula: bin = (lum[i] - lumMin) / lumRange * numBins
  unsigned int* d_histo = NULL;
  checkCudaErrors( cudaMalloc(&d_histo, numBins*sizeof(unsigned int)) );
  checkCudaErrors( cudaMemset(d_histo, 0, sizeof(unsigned int)*numBins));

  gen_histo<<< gridSize, blockSize, numBins*sizeof(int)>>>
      ( d_histo, d_logLuminance, min_logLum, range, numBins, numRows, numCols);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


  // 4) Perform an exclusive scan (prefix sum) on the histogram to get
  //    the cumulative distribution of luminance values (this should go in the
  //    incoming d_cdf pointer which already has been allocated for you)
  const dim3 gridES(1,1,1);
  const dim3 blockES((numBins/2),1,1);

  exclusive_scan<<<gridES, blockES, sizeof(unsigned int)*numBins>>>
      ( d_cdf, d_histo, numBins );

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_histo));

}
