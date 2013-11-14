//Udacity HW 4
//Radix Sorting

#include <cassert>
#include <cstring>
#include <cstdio>
#include <stdint.h>

#include <algorithm>

#include "reference_calc.h"
#include "utils.h"

//#include "quicksort.h"

#define MAX_THREADS 1024


/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

#if 1
/*
histo[0](118983,101497),  histo[1](110205,110275),
histo[2](110021,110459),  histo[3](109913,110567),
histo[4](110267,110213),  histo[5](110493,109987),
histo[6](110067,110413),  histo[7](109837,110643),
histo[8](110064,110416),  histo[9](110043,110437),
histo[10](111037,109443), histo[11](110788,109692),
histo[12](111193,109287), histo[13](111433,109047),
histo[14](111952,108528), histo[15](112241,108239),
histo[16](111609,108871), histo[17](100344,120136),
histo[18](100286,120194), histo[19](101878,118602),
histo[20](104156,116324), histo[21](105779,114701),
histo[22](109431,111049), histo[23](103261,117219),
histo[24](102942,117538), histo[25](117587,102893),
histo[26](9,220471), histo[27](1,220479),
histo[28](1,220479), histo[29](1,220479),
histo[30](220480,0), histo[31](220480,0)
*/


// generate a histogram for each bit order of the numSamps inputs
__global__ void gen_histo(unsigned int* d_out,
                          const unsigned int* const d_in,
                          const int numSamps)
{
    /*
      d_in is numSamps each with 32bits (so numSamps )
      d_out is 2 outputs for each bit (so 32 x 2 )

      numBits must equal gridDim.y
    */
    const int2 my2DPos = make_int2( (blockDim.x * blockIdx.x) + threadIdx.x,
                                    (blockDim.y * blockIdx.y) + threadIdx.y );
    // my memory location
    const int my1DPos = my2DPos.x;
    //const int numBits = gridDim.y;

    if( my1DPos >= numSamps )
        return;

    // 2 possible out bins for each of the bit orders
    // aka base, and max power
    const int2 numBins = make_int2( 2, blockIdx.y);

    // xbin is either 0 or 1 for all blocks
    const int xbin = ((d_in[my1DPos] & (0x1 << numBins.y)) == 0) ? 0 : 1;
    // ybin is blockIdx.y for all blocks
    //const int ybin = numBins.y;
    const int bin1DPos = xbin + (numBins.y * numBins.x);

    atomicAdd( &(d_out[bin1DPos]), 1);
}
#endif

__global__ void gen_pred(unsigned int* d_zeroOut, unsigned int* d_oneOut,
                         const unsigned int* const d_in,
                         const unsigned int numSamps, const unsigned int shift)
{
    // d_zeroOut is the predicate output for zeros
    // d_oneOut is the predicate output for the ones
    // d_in is the inputs
    // numSamps is the number of samples in the input,
    //    we will have numSamps output to each of the
    //    output arrays
    // shift is the bit specifier (which bit location of
    //    the input to check if is a 1 or a 0

    // my memory location
    const int myPos = (blockDim.x * blockIdx.x) + threadIdx.x;

    if( myPos >= numSamps )
        return;

    if( (d_in[myPos] & (0x1<<shift)) == 0 )
    {
        d_zeroOut[myPos] = 1;
        d_oneOut[myPos] = 0;
    }
    else
    {
        d_zeroOut[myPos] = 0;
        d_oneOut[myPos] = 1;
    }
}

#define LOAD_INPUTS 1
#define REDUCE_STEPS 1
#define SAVE_SUM 1
#define DOWNSWEEP_STEPS 1
#define SAVE_OUTPUTS 1

__global__ void exclusive_scan(unsigned int* d_out, unsigned int* d_sums,
                               const unsigned int* const d_in,
                               const int numSamps)
{
  // blelloch scan (reduce and downsweep)
  extern __shared__ unsigned int sdata[];

  const int tid = threadIdx.x;
  // block is sized for half the data (1 thread does 2 samps) so *2
  const int numBlkSamps = blockDim.x*2;
  const int myPos = threadIdx.x + (blockIdx.x * blockDim.x);

  int offset = 1;

  // zero out the shared memory
  //(this is required but only for the threads that return early)
  // yet go ahead and do for all
  sdata[2*tid] = 0;
  sdata[2*tid+1] = 0;
  __syncthreads();

  if( (2*myPos+1) >= numSamps )
  {
    //printf("returning early BlockIdx = (%d, %d, %d), BlockDim = (%d, %d, %d)\n",
    //       blockIdx.x, blockIdx.y, blockIdx.z,
    //       blockDim.x, blockDim.y, blockDim.z);

    return;
  }

#if LOAD_INPUTS
  // copy data into shared mem
  sdata[2*tid] = d_in[2*myPos];
  sdata[2*tid+1] = d_in[2*myPos+1];
  __syncthreads();

#if 0 //__CUDA_ARCH__ >= 200
// if( tid == 0 )
// {
//   printf("BlockIdx = (%d, %d, %d), BlockDim = (%d, %d, %d)\n",
//       blockIdx.x, blockIdx.y, blockIdx.z,
//       blockDim.x, blockDim.y, blockDim.z);
//}
  if( sdata[2*tid] > numSamps || sdata[2*tid+1] > numSamps )
  {
    printf("Input data(%u, %u) is larger than numSamps(%u)\n", d_in[2*myPos], d_in[2*myPos+1], numSamps);
    return;
  }
#endif
#endif

#if REDUCE_STEPS
  // reductions steps
  for(unsigned int s=(numBlkSamps>>1); s > 0; s >>= 1)
  {
    if(tid < s)
    {
      int a = offset*(2*tid+1)-1;
      int b = offset*(2*tid+2)-1;

      sdata[b] += sdata[a];
    }
    __syncthreads();

    offset <<= 1;
  }
#endif

#if SAVE_SUM
  // reduction step done, clear the last element after saving it to sums
  if( tid == 0 )
  {
    if( d_sums )
    {
      // save out the sum before clearing (it is inclusive)
      d_sums[blockIdx.x] = sdata[numBlkSamps-1];

      // we don't have to do anything special about the last block
      // (which may not have a complete block's worth of samples)
      // because we don't care about the last blocks's sum
      // with an exclusive scan
    }

#if 0 //__CUDA_ARCH__ >= 200
    //printf("d_sum[%d] = %u\n", blockIdx.x, d_sums[blockIdx.x]);

    if( sdata[numBlkSamps-1] > numSamps )
    {
      printf("Final block data(%u) is larger than numSamps(%u)\n", sdata[numBlkSamps-1], numSamps);
      return;
    }
#endif

    sdata[numBlkSamps-1] = 0;
  }
#endif

#if DOWNSWEEP_STEPS
  // downsweep steps
  for(unsigned int s=1; s < numBlkSamps; s *= 2)
  {
    offset >>= 1;
    if( tid < s )
    {
      int a = offset*(2*tid+1)-1;
      int b = offset*(2*tid+2)-1;

      unsigned int tmp = sdata[a];
      sdata[a] = sdata[b];
      sdata[b] += tmp;
    }
    __syncthreads();
  }
#endif

#if SAVE_OUTPUTS
  // copy to output
  d_out[2*myPos]   = sdata[2*tid];
  d_out[2*myPos+1] = sdata[2*tid+1];

#if 0 //__CUDA_ARCH__ >= 200
  if( sdata[2*tid] > numSamps || sdata[2*tid+1] > numSamps )
  {
    printf("output data(%u, %u) is larger than numSamps(%u)\n", d_out[2*tid], d_out[2*tid+1], numSamps);
    return;
  }
#endif

#endif
}

#define SLOW_GEN_SCAN 1

__global__ void exclusive_scan_gen( unsigned int* d_out,
                                    const unsigned int* const d_unused, const unsigned int* const d_in,
                                    const unsigned int numSamps)
{
  //extern __shared__ unsigned int shisto[];

  // my memory location
  //const int myPos = (blockDim.x * blockIdx.x) + threadIdx.x;
  const int tid = threadIdx.x;

  if( tid >= numSamps )
  { return; }

#if SLOW_GEN_SCAN
  if( tid == 0 )
  {
    d_out[0] = 0;
    for(int i=1; i < numSamps; ++i)
    {
      d_out[i] = d_in[i-1] + d_out[i-1];
    }
  }
#else
  d_out[tid] = d_in[tid];
#endif
}


__global__ void sum_2input( unsigned int* d_out,
                            const unsigned int* const d_in1, const unsigned int* const d_sums,
                            const unsigned int numSamps)
{
  extern __shared__ unsigned int sdata[];

  const int tid = threadIdx.x;
  const int myPos = threadIdx.x + (blockIdx.x * blockDim.x);

  if( myPos >= numSamps ) { return; }

  if( tid == 0 )
  {
    sdata[0] = d_sums[blockIdx.x];
  }
  __syncthreads();

  //d_out[myPos] = d_in1[myPos] + d_sums[blockIdx.x];
  d_out[myPos] = d_in1[myPos] + sdata[0];
}


#define LOAD_SH_SORT 1
#define LOAD_IN_SORT 1
#define LOAD_PD_SORT 1
#define SAVE_SORT 1

__global__ void sort( unsigned int* d_outV, unsigned int* d_outP,
                      const unsigned int* const d_inV, const unsigned int* const d_inP,
                      const unsigned int* const d_histo, const unsigned int* const d_1Pred,
                      const unsigned int* const d_1Scan, const unsigned int* const d_0Scan,
                      const unsigned int numSamps)
{
  extern __shared__ unsigned int shisto[];

  // my memory location
  const int myPos = (blockDim.x * blockIdx.x) + threadIdx.x;
  const int tid = threadIdx.x;

  if( myPos >= numSamps )
  { return; }

#if LOAD_SH_SORT
  // read in histo data
  if( tid == 0 )
  {
    shisto[0] = 0;
    shisto[1] = d_histo[0];
  }
  __syncthreads();
#endif

#if LOAD_IN_SORT
  // this thread's inputs
  const unsigned int inV = d_inV[myPos];
  const unsigned int inP = d_inP[myPos];
#else
  const unsigned int inV = 0;
  const unsigned int inP = 0;
#endif

#if LOAD_PD_SORT
  // this thread's predicate ( if ==0 is a 0 val, if ==1 is a 1 val)
  const unsigned int pred = d_1Pred[myPos];
#else
  const unsigned int pred = 0;
#endif


  // this thread's relative location in 0's output or 1's output
  const unsigned int relIdx = pred ? d_1Scan[myPos]: d_0Scan[myPos];

  // this thread's starting location
  const unsigned int startIdx = pred ? shisto[1] : shisto[0];

  // this thread's output location
  const unsigned int outIdx = startIdx + relIdx;


#if SAVE_SORT
  // write output to new location
  if( outIdx < numSamps )
  {
    d_outV[outIdx] = inV;
    d_outP[outIdx] = inP;
  }
#if __CUDA_ARCH__ >= 200
  else
  {
    printf("block(%d,%d,%d):thread(%d,%d,%d): OutIdx(%u) is too large(%u)! pred(%u) relIdx(%u) startIdx(%u) histo(%u, %u)\n",
      blockIdx.x,blockIdx.y,blockIdx.z,
      threadIdx.x, threadIdx.y, threadIdx.z,
      outIdx, numSamps,
      pred, relIdx, startIdx, shisto[0], shisto[1]);
  }
#endif

#endif
}


__global__ void swap( unsigned int* const d_outVals, unsigned int* const d_outPos,
                      unsigned int* const d_inVals, unsigned int* const d_inPos,
                      const unsigned int numSamps )
{
  // my memory location
  const int myPos = (blockDim.x * blockIdx.x) + threadIdx.x;
  //const int tid = threadIdx.x;

  if( myPos >= numSamps )
  { return; }

  unsigned int inV = d_inVals[myPos];
  unsigned int inP = d_inPos[myPos];

  d_inVals[myPos] = d_outVals[myPos];
  d_inPos[myPos]  = d_outPos[myPos];

  d_outVals[myPos] = inV;
  d_outPos[myPos]  = inP;
}

// gen_histo works, yay!!!
#define HISTO_REF 0
#if HISTO_REF
#include <fstream>
#endif

// preds seem to be ok
#define PRED_VALIDATE 0

// working looks like
#define SCAN_VALIDATE 0

// sums scan are ok
#define SUMS_VALIDATE 0

#define SORT_VALIDATE 0
#if SORT_VALIDATE
#include <sstream>
#include <fstream>
#endif

// do not use with other validate
#define CPU_VALIDATE 0
#if CPU_VALIDATE
#include <fstream>
#endif


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
  //TODO
  //PUT YOUR SORT HERE
{
  const unsigned int numPoss = 2; // number of possible outcomes
  const unsigned int numBits = 32; // number of bits

#if CPU_VALIDATE
  unsigned int* h_inputVals = NULL;
  unsigned int* h_inputPos = NULL;

  h_inputVals = new unsigned int[numElems];
  h_inputPos = new unsigned int[numElems];

  checkCudaErrors(cudaMemcpy(h_inputVals, d_inputVals, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_inputPos, d_inputPos, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
#endif


  
  // numElems must be a power of 2
  assert( (numElems%2) == 0 );

#if HISTO_REF
  unsigned int histo[numBits*numPoss];
  unsigned int dev_histo[numBits*numPoss];
  std::cout << "NumElems: " << numElems
            << ", sizeof(histo): " << sizeof(histo)
            << std::endl;

  memset( histo, 0, sizeof(histo) );
  unsigned int* h_inputVals = new unsigned int[numElems];
  checkCudaErrors(cudaMemcpy(h_inputVals, d_inputVals, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));

std::ofstream fout("inputVals.txt");
for(int i=0; i < numElems; ++i)
{
if( fout.good() ) fout << std::hex << h_inputVals[i] << std::endl;
else
  std::cout << "Cannot write to inputVals.txt!" << std::endl;
}
fout.close();

  for(int i=0; i < numElems; ++i)
  {
    for(int b=0; b < numBits; ++b)
    {
      if( (h_inputVals[i] & (1<<b)) == 0 )
        ++(histo[b*numPoss]);
      else
        ++(histo[b*numPoss+1]);
    }
  }
  //std::cout << "histo calculated" << std::endl;

  //for(int i=0; i < numBits; ++i)
  //{
  //    std::cout << "histo[" << i << "](" << histo[i*numPoss] << "," << histo[i*numPoss+1] << "), ";
  //}
  //std::cout << std::endl;
#endif

  /*
    uint2* posVals = new uint2[numElems];
    for(int i=0; i < numElems; ++i)
    {
    posVals[i].x = d_inputVals[i];
    posVals[i].y = d_inputPos[i];
    }
  */

  const int numThreads = (MAX_THREADS < numElems) ? MAX_THREADS : numElems;
  const dim3 blockSize_histo( numThreads, 1, 1);
  const dim3 gridSize_histo( (numElems + blockSize_histo.x-1) / blockSize_histo.x, numBits, 1);

  unsigned int* d_histo = NULL;
  checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int) * numPoss * numBits ));
  checkCudaErrors(cudaMemset(d_histo, 0,  sizeof(unsigned int) * numPoss*numBits));

  gen_histo<<<gridSize_histo, blockSize_histo>>>( d_histo, d_inputVals, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // we can execlusively scan the d_histo to put so the
  // follow-on functions can look at the same location
  // (e.g. 0's read d_histo[blockIdx][0] (always 0) and 1's read d_histo[blockIdx][1]),
  // but since only 2 values just as east to have 0's always write to 0
  // and have the 1's read d_histo[blockIdx][0];


#if HISTO_REF
  checkCudaErrors(cudaMemcpy(dev_histo, d_histo, sizeof(unsigned int) * numPoss * numBits, cudaMemcpyDeviceToHost));
  for(int i=0; i < numBits; ++i)
  {
    if( histo[i*numPoss] != dev_histo[i*numPoss] )
      std::cout << "histo[" << i << "](" << histo[i*numPoss] << "," << dev_histo[i*numPoss] << "), ";

    if( histo[i*numPoss+1] != dev_histo[i*numPoss+1] )
      std::cout << "histo[" << i << "](" << histo[i*numPoss+1] << "," << dev_histo[i*numPoss+1] << ")\n";
  }
  std::cout << std::endl;
#endif

  const dim3 blockSize_pred( numThreads, 1, 1);
  const dim3 gridSize_pred( (numElems + blockSize_pred.x-1) / blockSize_pred.x, 1, 1);

  unsigned int* d_zeroPred = NULL;
  unsigned int* d_onePred = NULL;
  checkCudaErrors(cudaMalloc(&d_zeroPred, sizeof(unsigned int) * numElems ));
  checkCudaErrors(cudaMalloc(&d_onePred, sizeof(unsigned int) * numElems ));


  // each thread in the scan does two elements so numThreads/2 with same gridSize
  // for half the number of threads as elements
  const dim3 blockSize_scan( (numThreads/2), 1, 1 );
  const dim3 gridSize_scan( (numElems + blockSize_pred.x-1) / blockSize_pred.x, 1, 1);

  // numThreads is a power of 2, ok
  //std::cout << "numThreads = " << numThreads << std::endl;

  unsigned int* d_zeroScan = NULL;
  unsigned int* d_oneScan  = NULL;
  unsigned int* d_zeroSums = NULL;
  unsigned int* d_oneSums  = NULL;
  unsigned int* d_1Sums = NULL;
  unsigned int* d_0Sums = NULL;

  // if there are an odd number of blocks set pad to 1 to we can specify an even number of blocks
  unsigned int pad = 0;
  if( gridSize_scan.x % 2 )
  {  ++pad; }
  assert( (gridSize_scan.x + pad) % 2 == 0 );

  checkCudaErrors(cudaMalloc(&d_zeroScan, sizeof(unsigned int) * numElems ));
  checkCudaErrors(cudaMalloc(&d_oneScan, sizeof(unsigned int) * numElems ));
  checkCudaErrors(cudaMalloc(&d_zeroSums, sizeof(unsigned int) * (gridSize_scan.x + pad)));
  checkCudaErrors(cudaMalloc(&d_oneSums, sizeof(unsigned int) * (gridSize_scan.x + pad)));
  checkCudaErrors(cudaMalloc(&d_0Sums, sizeof(unsigned int) * (gridSize_scan.x + pad)));
  checkCudaErrors(cudaMalloc(&d_1Sums, sizeof(unsigned int) * (gridSize_scan.x + pad)));

  //std::cout << "Scan gridSize_scan.x = " << gridSize_scan.x << ", pad = " << pad
  //<< ", blockSize_scan.x = " << blockSize_scan.x
  //<< std::endl;

  // we are using the bufferes to ping-pong back and forth
  // so input and outputs change each loop iterator
  // (setup backwards as the loop will swap before using)
  unsigned int *d_outV = d_inputVals;
  unsigned int *d_outP = d_inputPos;
  unsigned int *d_inV = d_outputVals;
  unsigned int *d_inP = d_outputPos;

  for(unsigned int bit=0; bit < numBits; ++bit)
  {
    //swap input and output pointers (we do it before so when
    // we leave the loop the pointers will point to their last
    // action (input or output)
    if( d_outV == d_outputVals )
    {
      d_outV = d_inputVals;
      d_outP = d_inputPos;
      d_inV = d_outputVals;
      d_inP = d_outputPos;
    }
    else
    {
      d_outV = d_outputVals;
      d_outP = d_outputPos;
      d_inV = d_inputVals;
      d_inP = d_inputPos;
    }

    // get predicates for if is a 1 or is a 0
    gen_pred<<<gridSize_pred, blockSize_pred>>>( d_zeroPred, d_onePred, d_inV, numElems, bit);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#if PRED_VALIDATE
    unsigned int* h_one = NULL;
    unsigned int* h_zero = NULL;
    unsigned int* h_inputVals = new unsigned int[numElems];

    h_one = new unsigned int[numElems];
    h_zero = new unsigned int[numElems];

    checkCudaErrors(cudaMemcpy(h_inputVals, d_inV, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_one, d_onePred, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_zero, d_zeroPred, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));

    std::cout << "Checking predicates for bit" << bit << std::endl;
    std::stringstream psstrm;
    psstrm << "preds" << bit << ".txt";
    std::ofstream pfout(psstrm.str().c_str());

    pfout << "Samp:  Pred0     Pred1      inputVals" << std::endl;
    for(int i=0; i < numElems; ++i)
    {
      pfout << std::dec << " " << i << "     " << h_zero[i] << "      " << h_one[i]
            << "     0x" << std::hex << h_inputVals[i]
            << std::endl;

      // std::cout << std::dec << "bit" << bit << ",samp" << i
      //           << ": h_zero = " << h_zero[i] << ", h_one = " << h_one[i]
      //           << ", inputVal = " << std::hex << h_inputVals[i]
      //           << std::endl;
       if( (h_inputVals[i] & (0x1<<bit)) == 0 )
       {
         if( !h_zero[i] )
           std::cout << "sample" << i << ", bit" << bit << ": is 0 but pred0 = " << h_zero[i] << std::endl;
       }
       else
       {
         if( !h_one[i] )
           std::cout << "sample" << i << ", bit" << bit << ": is 1 but pred1 = " << h_one[i] << std::endl;

       }
      if(h_one[i] != 0 && h_one[i] != 1)
      {
        std::cout << "OnePred[" << i << "] (" << h_one[i] << ") is not a 0 or 1!" << std::endl;
      }
      if(h_zero[i] != 0 && h_zero[i] != 1)
      {
        std::cout << "ZeroPred[" << i << "] (" << h_zero[i] << ") is not a 0 or 1!" << std::endl;
      }
      if( (h_zero[i]) != !(h_one[i]) )
        std::cout << "invalid: h_zero[" << i << "](" << h_zero << "), h_one[" << i << "](" << h_one[i] << ")" << std::endl;
    }
    delete [] h_inputVals; h_inputVals = NULL;
    delete [] h_one; h_one = NULL;
    delete [] h_zero; h_zero = NULL;
#endif

#if 1 //scan
    // exclusive scan the output of the predicate arrays
    // to get the relative offsets

    exclusive_scan<<<gridSize_scan, blockSize_scan, (blockSize_scan.x*2*sizeof(unsigned int))>>>
        (d_zeroScan, d_0Sums, //outputs
         d_zeroPred, //inputs
         numElems); // params
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   exclusive_scan<<<gridSize_scan, blockSize_scan, blockSize_scan.x*2*sizeof(unsigned int)>>>
        (d_oneScan, d_1Sums,
         d_onePred,
         numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#if SCAN_VALIDATE
    unsigned int* h_scan1 = NULL;
    unsigned int* h_scan0 = NULL;
    unsigned int* h_sums1 = NULL;
    unsigned int* h_sums0 = NULL;
    unsigned int* h_pred0 = NULL;
    unsigned int* h_pred1 = NULL;

    h_pred0 = new unsigned int[numElems];
    h_pred1 = new unsigned int[numElems];
    h_scan1 = new unsigned int[numElems];
    h_scan0 = new unsigned int[numElems];
    h_sums1 = new unsigned int[(gridSize_scan.x + pad)];
    h_sums0 = new unsigned int[(gridSize_scan.x + pad)];

    checkCudaErrors(cudaMemcpy(h_pred1, d_onePred, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_pred0, d_zeroPred, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_scan1, d_oneScan, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_scan0, d_zeroScan, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_sums0, d_0Sums, ((gridSize_scan.x + pad)*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_sums1, d_1Sums, ((gridSize_scan.x + pad)*sizeof(unsigned int)), cudaMemcpyDeviceToHost));



    // for(int i=0; i < (gridSize_scan.x + pad); ++i)
    // {
    //   std::cout << "bit" << bit << " sums" << i << " (" << h_sums0[i] << ", " << h_sums1[i] << ")" << std::endl;
    // }

    //std::cout << "Checking scans for bit" << bit << std::endl;
    for(int i=0; i < gridSize_scan.x; ++i)
    {
      for(int j=1; j < blockSize_scan.x; ++j)
      {
        const unsigned int idx = i*blockSize_scan.x + j;
        if( idx < numElems )
        {
          if( h_scan1[idx] > numElems)
          {
            std::cout << "idx = " << idx << ", Scan1[" << i << "][" << j
                      << "]( " << h_scan1[idx]
                      << ") is more than numElems(" << numElems << ")"
                      << std::endl;
          }
          if( h_scan0[idx] > numElems)
          {
            std::cout << "idx = " << idx << ", Scan0[" << i << "][" << j
                      << "]( " << h_scan0[idx]
                      << ") is more than numElems(" << numElems << ")"
                      << std::endl;
            return;
          }
        }
      }
    }

#endif //scan validate

#endif
    // we don't support recursive scans (if number of blocks of
    // the first scan is more than will fit into a block then
    // we will need to do the scan->sum step multiple times
    // and we don't do that yet).
    assert( gridSize_scan.x <= MAX_THREADS );

#if 1 //SUM_SCANS
    // now scan the sums (pad is needed in case was an odd number of blocks
    // as scan only supports multiple of 2 inputs since it does 2 per thread)
    // /2 then + 1 as the numThreads launched can be odd
    const dim3 blockSize_sums( (gridSize_scan.x/2 + pad), 1, 1);
    const dim3 gridSize_sums(1, 1, 1);

    //std::cout << "gridSize_sums = " << gridSize_sums.x
    //          << ", blockSize_sums = " << blockSize_sums.x
    //          << ", sharedSize = " << 2*blockSize_sums.x*2 << " | " << 2*blockSize_sums.x*2*sizeof(unsigned int)
    //          << "b, numSamps = " << (gridSize_scan.x + pad)
    //          << std::endl;

#if SUMS_VALIDATE
    unsigned int* h_sin1 = NULL;
    unsigned int* h_sin0 = NULL;
    unsigned int* h_sout1 = NULL;
    unsigned int* h_sout0 = NULL;

    h_sin1 = new unsigned int[(gridSize_scan.x + pad)];
    h_sin0 = new unsigned int[(gridSize_scan.x + pad)];
    h_sout1 = new unsigned int[(gridSize_scan.x + pad)];
    h_sout0 = new unsigned int[(gridSize_scan.x + pad)];

    checkCudaErrors(cudaMemcpy(h_sin0, d_0Sums, ((gridSize_scan.x + pad)*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_sin1, d_1Sums, ((gridSize_scan.x + pad)*sizeof(unsigned int)), cudaMemcpyDeviceToHost));

#endif

    // numSamps: +pad because numSamps must be multiple of 2
    exclusive_scan_gen<<< gridSize_sums, blockSize_sums, 2*blockSize_sums.x*2*sizeof(unsigned int)>>>
        ( d_oneSums, NULL, d_1Sums, (gridSize_scan.x+pad) );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    exclusive_scan_gen<<< gridSize_sums, blockSize_sums, 2*blockSize_sums.x*2*sizeof(unsigned int)>>>
        ( d_zeroSums, NULL, d_0Sums, (gridSize_scan.x+pad) );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


#if SUMS_VALIDATE
    checkCudaErrors(cudaMemcpy(h_sout0, d_zeroSums, ((gridSize_scan.x + pad)*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_sout1, d_oneSums, ((gridSize_scan.x + pad)*sizeof(unsigned int)), cudaMemcpyDeviceToHost));

    bool mismatch = false;
    unsigned int* h_sref0 = new unsigned int[(gridSize_scan.x+pad)];
    unsigned int* h_sref1 = new unsigned int[(gridSize_scan.x+pad)];
    h_sref0[0] = 0;
    h_sref1[0] = 0;
    for(int i=1; i < (gridSize_scan.x+pad); ++i )
    {
      h_sref0[i] = h_sref0[i-1] + h_sin0[i-1];
      h_sref1[i] = h_sref1[i-1] + h_sin1[i-1];
      //      std::cout << "h_sin(" << h_sin0[i-1] << "," << h_sin1[i-1]
      //          << ")   h_sref(" << h_sref0[i-1] << "," << h_sref1[i-1]
      //          << ")   h_sout(" << h_sout0[i-1] << "," << h_sout1[i-1] << ")"
      //          << std::endl;
    }

#if 1
    for(int i=1; i < (gridSize_scan.x + pad); ++i)
    {
      if( h_sref0[i] != h_sout0[i] )
      {
        mismatch = true;
        std::cout << "sample" << i << ": h_sref0 = " << h_sref0[i] << ", h_sout0 = " << h_sout0[i] << std::endl;
      }
      if( h_sref1[i] != h_sout1[i] )
      {
        mismatch = true;
        std::cout << "sample" << i << ": h_sref1 = " << h_sref1[i] << ", h_sout1 = " << h_sout1[i] << std::endl;
      }
      if( (h_sin0[i-1] + h_sout0[i-1]) != h_sout0[i] )
      {
        mismatch = true;
        std::cout << "bit" << bit << ", sumsScan0 incorrect: in[" << i << "] + out[" << i-1 << "]"
              << " != out[" << i << "] ( " << h_sin0[i] << " + " << h_sout0[i-1] << " != " << h_sout0[i]
              << std::endl;
      }
      if( (h_sin1[i-1] + h_sout1[i-1]) != h_sout1[i] )
      {
        mismatch = true;
        std::cout << "bit" << bit << ", sumsScan1 incorrect: in[" << i << "] + out[" << i-1 << "]"
              << " != out[" << i << "] ( " << h_sin1[i] << " + " << h_sout1[i-1] << " != " << h_sout1[i]
              << std::endl;
      }

    }
#endif
    delete [] h_sin0; h_sin0 = NULL;
    delete [] h_sin1; h_sin1 = NULL;
    delete [] h_sout0; h_sout0 = NULL;
    delete [] h_sout1; h_sout1 = NULL;

    if( mismatch ) return;
#endif //sums validate

#endif //scans

#if 1 //sums
    // then add the scaned sums to each previously scanned block
    sum_2input<<< gridSize_pred, blockSize_pred, sizeof(unsigned int) >>>
        ( d_zeroScan, d_zeroScan, d_zeroSums, numElems );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    sum_2input<<< gridSize_pred, blockSize_pred, sizeof(unsigned int) >>>
        ( d_oneScan, d_oneScan, d_oneSums, numElems );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#if SCAN_VALIDATE
    unsigned int* h_ref0 = new unsigned int[numElems];
    unsigned int* h_ref1 = new unsigned int[numElems];

    checkCudaErrors(cudaMemcpy(h_scan1, d_oneScan, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_scan0, d_zeroScan, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));

    // host scan
    h_ref0[0] = 0;
    h_ref1[0] = 0;
    for(int i=1; i < numElems; ++i)
    {
      h_ref0[i] = h_pred0[i-1] + h_ref0[i-1];
      h_ref1[i] = h_pred1[i-1] + h_ref1[i-1];
      //std::cout << "sample" << i << ": h_ref0 = " << h_ref0[i] << ", h_ref1 = " << h_ref1[i] << std::endl;
    }

    //std::cout << "Checking final scan for bit" << bit << std::endl;
    bool mismatch = false;
    for(int i=0; i < numElems; ++i)
    {
      if( h_ref0[i] != h_scan0[i] )
      {
        std::cout << "sample" << i << ": host ref0 sum(" << h_ref0[i] << ") != device (" << h_scan0[i] << ")" << std::endl;
        mismatch = true;
      }
      if( h_ref1[i] != h_scan1[i] )
      {
        std::cout << "sample" << i << ": host ref1 sum(" << h_ref1[i] << ") != device (" << h_scan1[i] << ")" << std::endl;
        mismatch = true;
      }
    }

    delete [] h_ref0;
    delete [] h_ref1;
    delete [] h_pred0; h_pred0 = NULL;
    delete [] h_pred1; h_pred1 = NULL;
    delete [] h_scan1; h_scan1 = NULL;
    delete [] h_scan0; h_scan0 = NULL;
    delete [] h_sums0; h_sums0 = NULL;
    delete [] h_sums1; h_sums1 = NULL;

    if( mismatch ) return;
#endif //scan validate

#endif
#if 1 //sort
    // d_(zero|one)Scan now contains the reletive addresses for this bit's values
    // d_histo contains the offsets to know where to start writing
    // combine histo output, scan sum, and predicate to copy vals and pos
    // to new locations
    // only 1 predicate array is needed as there are only 2 outcomes (0 or 1)
    // so onePred is 0 for the 0's and 1 for the 1's
    const dim3 blockSize_sort = blockSize_pred;
    const dim3 gridSize_sort = gridSize_pred;

    // each thread sorts 1 '1' and 1 '0'
    // see about having a 1 thread sort just 1 value
    //const unsigned int histoSize = sizeof(unsigned int) * numPoss*numBits;

    //std::cout << "Sorting for bit" << bit << std::endl;
    const unsigned int sdataSize = numPoss * sizeof(unsigned int);

    // histo has only 1 value of interest per bit so just pass that address
    sort<<< gridSize_sort, blockSize_sort, sdataSize>>>
        ( d_outV, d_outP,
          d_inV, d_inP, &d_histo[bit*numPoss], d_onePred, d_oneScan, d_zeroScan,
          numElems);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#if SORT_VALIDATE
    unsigned int* h_inV = new unsigned int[numElems];
    unsigned int* h_outV = new unsigned int[numElems];
    unsigned int* h_pred1 = new unsigned int[numElems];
    unsigned int* h_1scan = new unsigned int[numElems];
    unsigned int* h_0scan = new unsigned int[numElems];
    unsigned int* h_histo = new unsigned int[numPoss*numBits];

    checkCudaErrors(cudaMemcpy(h_inV, d_inV, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_outV, d_outV, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(h_pred1, d_onePred, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_0scan, d_zeroScan, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_1scan, d_oneScan, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(h_histo, d_histo, (numPoss*numBits*sizeof(unsigned int)), cudaMemcpyDeviceToHost));

    std::stringstream sstrm;
    sstrm << "sorts" << bit << ".txt";
    std::ofstream fout(sstrm.str().c_str());
    fout << "samp:  inputVal   outputVal   1pred     1Scan     0Scan    histo" << std::endl;
    for(int i=0; i < numElems; ++i)
    {
      fout << " " << i << std::hex << "    0x" << h_inV[i] << "  0x" << h_outV[i] << std::dec
           << "     " << h_pred1[i] << "      " << h_1scan[i] << "        " << h_0scan[i]
           << "     " << h_histo[bit*numPoss] << " " << h_histo[bit*numPoss + 1]
           << std::endl;
    }
    fout.close();

    delete [] h_1scan;
    delete [] h_0scan;
    delete [] h_pred1;
    delete [] h_inV;
    delete [] h_outV;
#endif

#endif // sort
  }

  // see if the outputs are where we want them
  if( d_outV != d_outputVals )
  {
    std::cout << "Swapping inputs and outputs." << std::endl;

    // need to swap
    swap<<< gridSize_pred, blockSize_pred >>>
        ( d_outputVals, d_outputPos,
          d_inputVals, d_inputPos,
          numElems );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  }

#if CPU_VALIDATE
  unsigned int* h_outputVals = NULL;
  unsigned int* h_outputPos = NULL;
  unsigned int* h_refVals = NULL;
  unsigned int* h_refPos = NULL;

  h_outputVals = new unsigned int[numElems];
  h_outputPos = new unsigned int[numElems];
  h_refVals = new unsigned int[numElems];
  h_refPos = new unsigned int[numElems];

  checkCudaErrors(cudaMemcpy(h_outputVals, d_outputVals, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_outputPos, d_outputPos, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
  //checkCudaErrors(cudaMemcpy(h_refVals, d_inputVals, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));
  //checkCudaErrors(cudaMemcpy(h_refPos, d_inputPos, (numElems*sizeof(unsigned int)), cudaMemcpyDeviceToHost));

  reference_calculation(h_inputVals, h_inputPos, h_refVals, h_refPos, numElems);

  std::ofstream fout("inouts.txt");

  quicksort::valpos_t* h_srtVP = new quicksort::valpos_t[numElems];
  quicksort::valsort( h_srtVP, h_inputVals, h_inputPos, numElems);

  std::sort( h_inputVals, h_inputVals + numElems );

  for(unsigned int i =0; i < numElems; ++i)
  {
    //if( fout.good() ) fout << i << " " << h_inputVals[i] << ":" << h_inputPos[i] << "  " << h_outputVals[i] << ":" << h_outputPos[i]
    if( fout.good() ) fout << i << " " << h_srtVP[i].val << ":" << h_srtVP[i].pos << "  " << h_outputVals[i] << ":" << h_outputPos[i]
                          << "  " << h_refVals[i] << ":" << h_refPos[i]
                          //                       << " " << h_refVals[i]
                           << std::endl;
    if( (h_refVals[i] != h_outputVals[i]) || (h_refPos[i] != h_outputPos[i]))
    {
      std::cout << "sample" << i << " mismatch: " << h_refVals[i] << ":" << h_refPos[i]
                << " and " << h_outputVals[i] << ":" << h_outputPos[i]
                << std::endl;
    }

    if( h_inputVals[i] != h_outputVals[i] )
    {
      std::cout << "sorted inputVals[" << i << "]( " << h_inputVals[i]
                << ") != h_outputVals[" << i << "]( " << h_outputVals[i] << ")" << std::endl;
    }
    /*
    if( h_inputPos[i] != h_outputPos[i] )
    {
      std::cout << "sorted inputPos[" << i << "]( " << h_inputPos[i]
                << ") != h_outputPos[" << i << "]( " << h_outputPos[i] << ")" << std::endl;
    }
    */
  }
  fout.close();

  delete [] h_refVals; h_refVals = NULL;
  delete [] h_refPos; h_refPos = NULL;
  delete [] h_srtVP;
  delete [] h_outputVals;
  delete [] h_outputPos;
  delete [] h_inputVals;
  delete [] h_inputPos;
  h_outputVals= NULL;
  h_outputPos = NULL;
  h_inputVals = NULL;
  h_inputPos  = NULL;
#endif



  checkCudaErrors(cudaFree(d_histo));

  checkCudaErrors(cudaFree(d_zeroPred));
  checkCudaErrors(cudaFree(d_onePred));

  checkCudaErrors(cudaFree(d_zeroScan));
  checkCudaErrors(cudaFree(d_oneScan));
  checkCudaErrors(cudaFree(d_zeroSums));
  checkCudaErrors(cudaFree(d_oneSums));
  checkCudaErrors(cudaFree(d_0Sums));
  checkCudaErrors(cudaFree(d_1Sums));

  //delete [] posVals;
}
