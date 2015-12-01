//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */


#include <cstdio>
#include <iostream>
#include <string>
#include <algorithm>

#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.h"

#if 0
#define REF_MASK 1
#define MY_MASK 1
#define COMP_MASK (MY_MASK && REF_MASK && 1)

#define REF_BORI (REF_MASK && 1)
#define MY_BORI 1
#define COMP_BORI (MY_BORI && REF_BORI)
#define IS_EXTERIOR 0
#define IS_INTERIOR 2
#define IS_BORDER 1
#define TO_COPY IS_BORDER
#define TO_LEAVE IS_EXTERIOR

#define MY_GUESS_ 1
#define COMP_GUESS (MY_GUESS && 1)

#define REF_JACOBI1 1
#define MY_JACOBI1 1
#define COMP_JACOBI1 (REF_JACOBI1 && MY_JACOBI1 && 1)

#define REF_JACOBI2 1
#define MY_JACOBI2 1
#define COMP_JACOBI2 (REF_JACOBI2 && MY_JACOBI2 && 1)


__global__ void genMask( const uchar4* d_srcImg, uchar4* d_mask,
			 const uint numRows, const uint numCols )
{
  const uint numSamps = numRows*numCols;

  const uint pos = threadIdx.x + blockDim.x * blockIdx.x;
  if( pos < numSamps )
  {
    // first copy the input to the output
    d_mask[pos] = d_srcImg[pos];

    // no sync necessary as each thread only works on its own sample
    // next figure out the mask
      d_mask[pos].w = ( (d_mask[pos].x + d_mask[pos].y + d_mask[pos].z) < (3 * 255) ) ? TO_COPY : TO_LEAVE;

      /*
      d_mask[pos].w = ((d_mask[pos].x == 255) &&
		       (d_mask[pos].y == 255) &&
		       (d_mask[pos].z == 255)) ? 1 : 0;
      */
    }
}

__global__ void genBOrI( uchar4* d_mask,
                         const int nRows,
                         const int nCols )
{
  const uint nSamps = nRows*nCols;

  const uint samp = threadIdx.x + blockDim.x * blockIdx.x;
  if( (samp < nSamps) && (d_mask[samp].w) )
  {
    // reference: samp = myRow * nCols + myCol;
    int myRow = samp / nCols; //works due to integer math truncation
    int myCol = samp - myRow * nCols;

    assert( myCol < nCols );
    assert( myRow < nRows );
    assert( samp == (myRow * nCols + myCol));

    int upRow = myRow - 1;
    int downRow = myRow + 1;
    int leftRow = myRow;
    int rightRow = myRow;

    int leftCol = myCol - 1;
    int rightCol = myCol + 1;
    int upCol = myCol;
    int downCol = myCol;

    bool doUp = true,
         doDown = true,
         doLeft = true,
         doRight = true;
    if( upRow < 0 ) doUp = false;
    if( downRow > nRows ) doDown = false;
    if( leftCol < 0 ) doLeft = false;
    if( rightCol > nCols ) doRight = false;

    int upSamp = upRow * nCols + upCol;
    int downSamp = downRow * nCols + downCol;
    int leftSamp = leftRow * nCols + leftCol;
    int rightSamp = rightRow * nCols + rightCol;

    assert( upSamp <= nSamps );
    assert( downSamp <= nSamps );
    assert( leftSamp <= nSamps );
    assert( rightSamp <= nSamps );

    // .w == 1 if to be copied (interior or border )
    int upVal = ( doUp && (d_mask[ upSamp ].w != 0) ) ? 1 : 0;
    int downVal = ( doDown && (d_mask[ downSamp ].w != 0) ) ? 1 : 0;
    int leftVal = ( doLeft && (d_mask[ leftSamp ].w != 0) ) ? 1 : 0;
    int rightVal = ( doRight && (d_mask[ rightSamp ].w != 0 ) ) ? 1 : 0;

    int bOrI = 0;
    bOrI += upVal;
    bOrI += downVal;
    bOrI += leftVal;
    bOrI += rightVal;

  //  std::cout << "bori samp[" << samp << "] = " << bOrI << std::endl;

    assert( d_mask[samp].w == TO_COPY );
    assert( bOrI <= 4 );
    if( bOrI == 4 )
    {
      d_mask[samp].w = IS_INTERIOR;
    }
  }
}


__global__ void initGuessBuffers( const uchar4* srcImg,
                                  float3* guess1, float3* guess2,
                                  const uint nRows, const uint nCols )
{
  const uint nSamps = nRows*nCols;

  const uint samp = threadIdx.x + blockDim.x * blockIdx.x;
  if( samp < nSamps )
  {
    guess1[samp].x = srcImg[samp].x;
    guess2[samp].x = srcImg[samp].x;

    guess1[samp].y = srcImg[samp].y;
    guess2[samp].y = srcImg[samp].y;

    guess1[samp].z = srcImg[samp].z;
    guess2[samp].z = srcImg[samp].z;
  }
}

__host__ __device__ const float3 to_float3( const uchar4& rhs )
{
  float3 lhs;
  lhs.x = (float)rhs.x;
  lhs.y = (float)rhs.y;
  lhs.z = (float)rhs.z;
  return lhs;
}

__host__ __device__ const float3 to_float3( const float& rhs )
{
  float3 lhs;
  lhs.x = rhs;
  lhs.y = rhs;
  lhs.z = rhs;
  return lhs;
}

__host__ __device__ const float3 operator/(const float3& lhs, const float3& rhs)
{
  float3 rslt;
  rslt.x = lhs.x / rhs.x;
  rslt.y = lhs.y / rhs.y;
  rslt.z = lhs.z / rhs.z;
  return rslt;
}
__host__ __device__ float3& operator/=(float3& lhs, const float3& rhs)
{
  lhs.x /= rhs.x;
  lhs.y /= rhs.y;
  lhs.z /= rhs.z;
  return lhs;
}

__host__ __device__ const float3 operator*(const float3& lhs, const float3& rhs)
{
  float3 rslt;
  rslt.x = lhs.x * rhs.x;
  rslt.y = lhs.y * rhs.y;
  rslt.z = lhs.z * rhs.z;
  return rslt;
}
__host__ __device__ float3& operator*=(float3& lhs, const float3& rhs)
{
  lhs.x *= rhs.x;
  lhs.y *= rhs.y;
  lhs.z *= rhs.z;
  return lhs;
}

__host__ __device__ const float3 operator+(const float3& lhs, const float3& rhs)
{
  float3 rslt;
  rslt.x = lhs.x + rhs.x;
  rslt.y = lhs.y + rhs.y;
  rslt.z = lhs.z + rhs.z;
  return rslt;
}
__host__ __device__ float3& operator+=(float3& lhs, const float3& rhs)
{
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  lhs.z += rhs.z;
  return lhs;
}

__host__ __device__ const float3 operator-(const float3& lhs, const float3& rhs)
{
  float3 rslt;
  rslt.x = lhs.x - rhs.x;
  rslt.y = lhs.y - rhs.y;
  rslt.z = lhs.z - rhs.z;
  return rslt;
}
__host__ __device__ float3& operator-=(float3& lhs, const float3& rhs)
{
  lhs.x -= rhs.x;
  lhs.y -= rhs.y;
  lhs.z -= rhs.z;
  return lhs;
}


// computeG calculates the sum2 values.
//
//   Sum2: += (SourceImg[p] - SourceImg[neighbor])   (for all four neighbors)
//
//  Thus we have:
//          sum2 = (src[p] - src[n1]) + (src[p] - src[n2]) + (src[p] - src[n3]) + (src[p] - src[n4]);
//               = src[p] + src[p] + src[p] + src[p] - src[n1] - src[n2] - src[n3] - src[n4];
//               = 4*src[p] - src[n1] - src[n2] - src[n3] - src[n4];
//               = 4*src[p] - (src[n1] + src[n2] + src[n3] + src[n4]);
//
// mask is a uchar4 because it contains the source img rbg channels
// as well as the interior/border information (in .w)
//
__global__ void computeG( const uchar4* mask,
                          float3 *g,
                          const uint nRows, const uint nCols )
{
  const uint nSamps = nRows*nCols;
  const float3 z = make_float3( 0.f, 0.f, 0.f );


  const int samp = threadIdx.x + blockDim.x * blockIdx.x;
  if( samp < nSamps )
  {
    if( mask[samp].w == IS_INTERIOR )
    {
      // calc 4*src[p]
      float3 sum = make_float3( 4.f * mask[samp].x,
                                4.f * mask[samp].y,
                                4.f * mask[samp].z);

      // figure out our neighbor's positions, if valid use, otherwise use 0
      // left
      float3 n1 = ((samp-1) < 0) ? z : to_float3( mask[samp - 1]);
      // right
      float3 n2 = ((samp+1) < nSamps) ? to_float3( mask[samp + 1]) : z;
      // above
      float3 n3 = ((samp+(int)nCols) < nSamps) ? to_float3( mask[samp + nCols]) : z;
      // below
      float3 n4 = ((samp-(int)nCols) < 0) ? z : to_float3( mask[samp - nCols]);

      sum -= (n1 + n2 + n3 + n4);

      g[samp] = sum;
    }
    else
    {
      g[samp] = z;
    }
  }
}

#if REF_JACOBI1
// pre-compute the values of g, which depend only on the source image
// and aren't iteration dependent.
void refComputeG(const uchar4* const src,
                 float3* const g,
                 const size_t numRows,
                 const size_t numCols,
                 const uchar4* const interiorPixelList)
{
  const size_t numSamps = numRows * numCols;
  for(size_t i = 0; i < numSamps; ++i)
  {
    if( interiorPixelList[i].w == IS_INTERIOR )
    {
      float sumx = 4.f * src[i].x;
      float sumy = 4.f * src[i].y;
      float sumz = 4.f * src[i].z;

      const int left = i-1;
      const int right = i+1;
      const int above = i - numCols;
      const int below = i + numCols;

      if( left >= 0 )
      {
        sumx -= (float)src[left].x;
        sumy -= (float)src[left].y;
        sumz -= (float)src[left].z;
      }

      //FIXME: this should be clamped at (trunc(i/numRows) + numCols) >= i%numRows
      if( right <= numSamps )
      {
        sumx -= (float)src[right].x;
        sumy -= (float)src[right].y;
        sumz -= (float)src[right].z;
      }

      if( above >= 0 )
      {
        sumx -= (float)src[above].x;
        sumy -= (float)src[above].y;
        sumz -= (float)src[above].z;
      }

      // FIXME: clamp should not be on numSamps
      if( below <= numSamps)
      {
        sumx -= (float)src[below].x;
        sumy -= (float)src[below].y;
        sumz -= (float)src[below].z;
      }

      g[i].x = sumx;
      g[i].y = sumy;
      g[i].z = sumz;
    }
  }
}

#endif

#if REF_JACOBI2
//Performs one iteration of the solver
void computeIteration(const uchar4* const dstImg,
                      const uchar4* const mask,
                      const size_t numRows,
                      const size_t numCols,
                      const float3* const f,
                      const float3* const g,
                      float3* const f_next)
{
  const size_t numSamps = numRows * numCols;

  for (size_t i = 0; i < numSamps; ++i)
  {
    float3 sum  = make_float3(0.f, 0.f, 0.f);

    //process all 4 neighbor pixels
    //for each pixel if it is an interior pixel
    //then we add the previous f, otherwise if it is a
    //border pixel then we add the value of the destination
    //image at the border.  These border values are our boundary
    //conditions.
    const int left = i-1;
    const int right = i+1;
    const int above = i - numCols;
    const int below = i + numCols;

    if( left >= 0 )
      sum += (mask[left].w == IS_INTERIOR ) ? f[left] : to_float3(dstImg[left]);

    if( right <= numSamps )
      sum += (mask[right].w == IS_INTERIOR ) ? f[right] : to_float3(dstImg[right]);

    if( above >= 0 )
      sum += (mask[above].w == IS_INTERIOR ) ? f[above] : to_float3(dstImg[above]);

    if( below <= numSamps )
      sum += (mask[below].w == IS_INTERIOR ) ? f[below] : to_float3(dstImg[below]);

    float3 f_next_val = (sum + g[i]) / to_float3(4.f);

    f_next[i].x = std::min(255.f, std::max(0.f, f_next_val.x)); //clip to [0, 255]
    f_next[i].y = std::min(255.f, std::max(0.f, f_next_val.y)); //clip to [0, 255]
    f_next[i].z = std::min(255.f, std::max(0.f, f_next_val.z)); //clip to [0, 255]
  }

}
#endif

#if REF_BORI
std::string GetBOrIStr( const int _bori )
{
  switch( _bori )
  {
  case IS_INTERIOR:
    return std::string("interior");

  case IS_BORDER:
    return std::string("border");

  default:
    return std::string("exterior");
  }
}

int IsInteriorOrBorder( const size_t nRows,
                        const size_t nCols,
                        const size_t samp,
                        const unsigned char* const data )
{
  size_t nSamps = nRows * nCols;
  if ( samp >= nSamps )
    return false;

  // if data[samp] == 0, return early
  if( !data[samp] )
    return IS_EXTERIOR;

  // reference: samp = myRow * nCols + myCol;
  int myRow = samp / nCols; //works due to integer math truncation
  int myCol = samp - myRow * nCols;

  assert( myCol < nCols );
  assert( myRow < nRows );
  assert( samp == (myRow * nCols + myCol));

  int upRow = myRow - 1;
  int downRow = myRow + 1;
  int leftRow = myRow;
  int rightRow = myRow;

  int leftCol = myCol - 1;
  int rightCol = myCol + 1;
  int upCol = myCol;
  int downCol = myCol;

  bool doUp = true,
       doDown = true,
       doLeft = true,
       doRight = true;
  if( upRow < 0 ) doUp = false;
  if( downRow > nRows ) doDown = false;
  if( leftCol < 0 ) doLeft = false;
  if( rightCol > nCols ) doRight = false;

  size_t upSamp = upRow * nCols + upCol;
  size_t downSamp = downRow * nCols + downCol;
  size_t leftSamp = leftRow * nCols + leftCol;
  size_t rightSamp = rightRow * nCols + rightCol;

  assert( upSamp <= nSamps );
  assert( downSamp <= nSamps );
  assert( leftSamp <= nSamps );
  assert( rightSamp <= nSamps );

  // .w == 1 if to be copied (interior or border )
  size_t upVal = (doUp && data[ upSamp ]) ? 1 : 0;
  size_t downVal = (doDown && data[ downSamp ]) ? 1 : 0;
  size_t leftVal = (doLeft && data[ leftSamp ]) ? 1 : 0;
  size_t rightVal = (doRight && data[ rightSamp ]) ? 1 : 0;

  size_t bOrI = 0;
  bOrI += upVal;
  bOrI += downVal;
  bOrI += leftVal;
  bOrI += rightVal;

//  std::cout << "bori samp[" << samp << "] = " << bOrI << std::endl;

  assert( bOrI <= 4 );
  int result = IS_BORDER;
  if( bOrI == 4 )
  {
    result = IS_INTERIOR;
  }

  return result;
}
#endif

#endif //if 0

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
}

// take this out and the closing braket above
#if 0
  const size_t srcSize = numRowsSource * numColsSource;

  /* To Recap here are the steps you need to implement */
    /*
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.
    */
  uchar4* d_srcImg = NULL;
  uchar4* d_destImg = NULL;

  //first create mask
#if REF_MASK
  unsigned char* mask = new unsigned char[srcSize];

  for (int i = 0; i < srcSize; ++i) {
    mask[i] = ( (h_sourceImg[i].x + h_sourceImg[i].y + h_sourceImg[i].z) < (3 * 255) ) ? 1 : 0;
  }

#endif

  dim3 bSizeMask( 512, 1, 1);
  dim3 gSizeMask( (srcSize + bSizeMask.x - 1) / bSizeMask.x, 1, 1);

//  std::cout << "Grid: " << gSizeMask.x << ", " << gSizeMask.y << ", " << gSizeMask.z << std::endl;
//  std::cout << "Block: " << bSizeMask.x << ", " << bSizeMask.y << ", " << bSizeMask.z << std::endl;


#if MY_MASK
  uchar4* d_mask = NULL;
  checkCudaErrors( cudaMalloc(&d_mask, sizeof(uchar4)*srcSize) );
  checkCudaErrors( cudaMalloc(&d_srcImg, sizeof(uchar4)*srcSize) );

  checkCudaErrors(cudaMemcpy(d_srcImg, h_sourceImg, (sizeof(uchar4)*srcSize), cudaMemcpyHostToDevice));
    
  genMask<<< gSizeMask, bSizeMask >>>( d_srcImg, d_mask, numRowsSource, numColsSource );
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


#endif //end MY_MASK
#if COMP_MASK
  uchar4* h_mask = new uchar4[srcSize];
  checkCudaErrors(cudaMemcpy(h_mask, d_mask, (sizeof(uchar4)*srcSize), cudaMemcpyDeviceToHost));

  for( int i=0; i < srcSize; ++i)
  {
    if( h_mask[i].w != mask[i] )
    {
      std::cout << i << ": Mask Ref = (" << (unsigned int)mask[i]
                << "), Mask Calc = (" << (unsigned int)h_mask[i].w << ")"
                << std::endl;
    }
  }

  delete [] h_mask;
#endif //end COMP_MASK

  /*
    2) Compute the interior and border regions of the mask.  An interior
    pixel has all 4 neighbors also inside the mask.  A border pixel is
    in the mask itself, but has at least one neighbor that isn't.
  */

#if REF_BORI
  unsigned int *bori = new unsigned int[srcSize];
  for (size_t i = 0; i < srcSize; ++i)
  {
    bori[i] = IsInteriorOrBorder( numRowsSource,
                                  numColsSource,
                                  i,
                                  mask );

    //if( bori[i] != IS_EXTERIOR )
    //  std::cout << "Mask(" << static_cast<unsigned int>(mask[i]) << "), Sample " << i << " is " << GetBOrIStr(bori[i]) << std::endl;
  }
#endif
#if MY_BORI

  genBOrI<<< gSizeMask, bSizeMask >>>( d_mask, numRowsSource, numColsSource );
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#endif
#if COMP_BORI
  uchar4* h_bori = new uchar4[srcSize];
  checkCudaErrors(cudaMemcpy(h_bori, d_mask, (sizeof(uchar4)*srcSize), cudaMemcpyDeviceToHost));

  for( int i=0; i < srcSize; ++i)
  {
    if( h_bori[i].w != bori[i] )
    {
      std::cout << i << ": BOrI Ref = (" << static_cast<unsigned int>(bori[i])
          << "), BOrI Calc = (" << static_cast<unsigned int>(h_bori[i].w) << ")"
          << std::endl;
    }
  }

  delete [] h_bori;
#endif

    /* NOT DOING
     3) Separate out the incoming image into three separate channels
       NOT DOING
     */
    /*
     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.
     */
#if MY_GUESS
  float3* d_guess1 = NULL;
  float3* d_guess2 = NULL;
  checkCudaErrors( cudaMalloc(&d_guess1, sizeof(float3)*srcSize) );
  checkCudaErrors( cudaMalloc(&d_guess2, sizeof(float3)*srcSize) );

  initGuessBuffers<<< gSizeMask, bSizeMask >>>( d_srcImg, d_guess1, d_guess2, numRowsSource, numColsSource );
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#endif
#if COMP_GUESS
  float3* h_g1 = new float3[srcSize];
  float3* h_g2 = new float3[srcSize];

  checkCudaErrors(cudaMemcpy(h_g1, d_guess1, (sizeof(float3)*srcSize), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_g2, d_guess2, (sizeof(float3)*srcSize), cudaMemcpyDeviceToHost));

  for( int i=0; i < srcSize; ++i)
  {
    if( (reinterpret_cast<uchar>(h_g1[i].x) != h_sourceImg[i].x)
        || (reinterpret_cast<uchar>(h_g1[i].y) != h_sourceImg[i].y)
        || (reinterpret_cast<uchar>(h_g1[i].z) != h_sourceImg[i].z) )
    {
      std::cout << i
          << ": Src = (" << static_cast<uint>(h_sourceImg[i].x)
          << ", " << static_cast<uint>(h_sourceImg[i].y)
          << ", " << static_cast<uint>(h_sourceImg[i].z)
          << "), G1 = (" << static_cast<uint>(h_g1[i].x)
          << ", " << static_cast<uint>(h_g1[i].y) << ", "
          << ", " << static_cast<uint>(h_g1[i].z) << ", "
          << ")" << std::endl;
    }

    if( (reinterpret_cast<uchar>(h_g2[i].x) != h_sourceImg[i].x)
        || (reinterpret_cast<uchar>(h_g2[i].y) != h_sourceImg[i].y)
        || (reinterpret_cast<uchar>(h_g2[i].z) != h_sourceImg[i].z) )
    {
      std::cout << i
          << ": Src = (" << static_cast<uint>(h_sourceImg[i].x)
          << ", " << static_cast<uint>(h_sourceImg[i].y)
          << ", " << static_cast<uint>(h_sourceImg[i].z)
          << "), G1 = (" << static_cast<uint>(h_g2[i].x)
          << ", " << static_cast<uint>(h_g2[i].y) << ", "
          << ", " << static_cast<uint>(h_g2[i].z) << ", "
          << ")" << std::endl;
    }
  }

  delete [] h_g1;
  delete [] h_g2
#endif

    /*
     5) For each color channel perform the Jacobi iteration described 
        above 800 times.
    */
#if REF_JACOBI1
  /*
  Our initial guess is going to be the source image itself.  This is a pretty
  good guess for what the blended image will look like and it means that
  we won't have to do as many iterations compared to if we had started far
  from the final solution.

  ImageGuess_prev (Floating point)
  ImageGuess_next (Floating point)

  DestinationImg
  SourceImg

  Follow these steps to implement one iteration:

  1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
     Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
            else if the neighbor in on the border then += DestinationImg[neighbor]

     Sum2: += (SourceImg[p] - SourceImg[neighbor])   (for all four neighbors)

  2) Calculate the new pixel value:
     float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
     ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]
   */

  //next we'll precompute the g term - it never changes, no need to recompute every iteration
  float3 *g   = new float3[srcSize];
  uchar4* h_maskRef = new uchar4[srcSize];
  checkCudaErrors(cudaMemcpy(h_maskRef, d_mask, (sizeof(uchar4)*srcSize), cudaMemcpyDeviceToHost));


  memset(g, 0, srcSize * sizeof(float3));

  refComputeG(h_sourceImg, g, numRowsSource, numColsSource, h_maskRef);

  delete [] h_maskRef;
  h_maskRef = NULL;


#endif
#if MY_JACOBI1
  //pre-compute sum2 as it only uses the source.
  float3* d_g = NULL;
  checkCudaErrors( cudaMalloc(&d_g, sizeof(float3)*srcSize) );

  computeG<<< gSizeMask, bSizeMask >>>( d_mask, d_g, numRowsSource, numColsSource );
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#endif
#if COMP_JACOBI1
  float3* h_g = new float3[srcSize];
  checkCudaErrors(cudaMemcpy(h_g, d_g, (sizeof(float3)*srcSize), cudaMemcpyDeviceToHost));

  std::cout << "NumSamps: " << srcSize << ", NumRows: " << numRowsSource << ", NumCols: " << numColsSource << std::endl;
  for(int i = 0; i < srcSize; ++i)
  {
    if( (h_g[i].x != g[i].x) || (h_g[i].y != g[i].y) || (h_g[i].z != g[i].z) )
    {
      std::cout << i << ": RefG (" << g[i].x << ", " << g[i].y << ", " << g[i].z
                << "), CalcG (" << h_g[i].x << ", " << h_g[i].y << ", " << h_g[i].z << ")" << std::endl;
    }
  }

  delete [] h_g;
  h_g = NULL;
#endif

#if REF_JACOBI2
  float3* g1 = new float3[srcSize];
  float3* g2 = new float3[srcSize];
  float3* gM = new float3[srcSize];

  checkCudaErrors(cudaMemcpy(gM, d_g, (sizeof(float3)*srcSize), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(g1, d_guess1, (sizeof(float3)*srcSize), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(g2, d_guess2, (sizeof(float3)*srcSize), cudaMemcpyDeviceToHost));

  // 800 iterations
  for( int i=0; i < 800; ++i)
  {
    //Performs one iteration of the solver
    void computeIteration(const uchar4* const dstImg,
                        const uchar4* const mask,
                        const size_t numRowsSource,
                        const size_t numColsSource,
                        const float3* const g1,
                        const float3* const gM,
                        float3* const g2)

    // swap the pointers for the next iteration
    float3* tmp = g1;
    g1 = g2;
    g2 = tmp;
  }

#endif
#if MY_JACOBI2
#endif
#if COMP_JACOBI2
  float3* h_g1 = new float3[srcSize];
  float3* h_g2 = new float3[srcSize];

  checkCudaErrors(cudaMemcpy(h_g1, d_guess1, (sizeof(float3)*srcSize), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_g2, d_guess2, (sizeof(float3)*srcSize), cudaMemcpyDeviceToHost));

  //TODO:

  delete [] h_g1;
  delete [] h_g2;
  h_g1 = NULL;
  h_g2 = NULL;
#endif

    /*
     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.
    */
    
    /*
      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */


#if REF
  /*
void reference_calc(const uchar4* const h_sourceImg,
                    const size_t numRowsSource, const size_t numColsSource,
                    const uchar4* const h_destImg,
                    uchar4* const h_blendedImg){
  */
  //we need to create a list of border pixels and interior pixels
  //this is a conceptually simple implementation, not a particularly efficient one...

  //first create mask
  size_t srcSize = numRowsSource * numColsSource;
  unsigned char* mask = new unsigned char[srcSize];

  for (int i = 0; i < srcSize; ++i) {
    mask[i] = (h_sourceImg[i].x + h_sourceImg[i].y + h_sourceImg[i].z < 3 * 255) ? 1 : 0;
  }

  //next compute strictly interior pixels and border pixels
  unsigned char *borderPixels = new unsigned char[srcSize];
  unsigned char *strictInteriorPixels = new unsigned char[srcSize];

  std::vector<uint2> interiorPixelList;

  //the source region in the homework isn't near an image boundary, so we can
  //simplify the conditionals a little...
  for (size_t r = 1; r < numRowsSource - 1; ++r) {
    for (size_t c = 1; c < numColsSource - 1; ++c) {
      if (mask[r * numColsSource + c]) {
        if (mask[(r -1) * numColsSource + c] && mask[(r + 1) * numColsSource + c] &&
            mask[r * numColsSource + c - 1] && mask[r * numColsSource + c + 1]) {
          strictInteriorPixels[r * numColsSource + c] = 1;
          borderPixels[r * numColsSource + c] = 0;
          interiorPixelList.push_back(make_uint2(r, c));
        }
        else {
          strictInteriorPixels[r * numColsSource + c] = 0;
          borderPixels[r * numColsSource + c] = 1;
        }
      }
      else {
          strictInteriorPixels[r * numColsSource + c] = 0;
          borderPixels[r * numColsSource + c] = 0;

      }
    }
  }

  //split the source and destination images into their respective
  //channels
  unsigned char* red_src   = new unsigned char[srcSize];
  unsigned char* blue_src  = new unsigned char[srcSize];
  unsigned char* green_src = new unsigned char[srcSize];

  for (int i = 0; i < srcSize; ++i) {
    red_src[i]   = h_sourceImg[i].x;
    blue_src[i]  = h_sourceImg[i].y;
    green_src[i] = h_sourceImg[i].z;
  }

  unsigned char* red_dst   = new unsigned char[srcSize];
  unsigned char* blue_dst  = new unsigned char[srcSize];
  unsigned char* green_dst = new unsigned char[srcSize];

  for (int i = 0; i < srcSize; ++i) {
    red_dst[i]   = h_destImg[i].x;
    blue_dst[i]  = h_destImg[i].y;
    green_dst[i] = h_destImg[i].z;
  }

  //next we'll precompute the g term - it never changes, no need to recompute every iteration
  float *g_red   = new float[srcSize];
  float *g_blue  = new float[srcSize];
  float *g_green = new float[srcSize];

  memset(g_red,   0, srcSize * sizeof(float));
  memset(g_blue,  0, srcSize * sizeof(float));
  memset(g_green, 0, srcSize * sizeof(float));

  computeG(red_src,   g_red,   numColsSource, interiorPixelList);
  computeG(blue_src,  g_blue,  numColsSource, interiorPixelList);
  computeG(green_src, g_green, numColsSource, interiorPixelList);

  //for each color channel we'll need two buffers and we'll ping-pong between them
  float *blendedValsRed_1 = new float[srcSize];
  float *blendedValsRed_2 = new float[srcSize];

  float *blendedValsBlue_1 = new float[srcSize];
  float *blendedValsBlue_2 = new float[srcSize];

  float *blendedValsGreen_1 = new float[srcSize];
  float *blendedValsGreen_2 = new float[srcSize];

  //IC is the source image, copy over
  for (size_t i = 0; i < srcSize; ++i) {
    blendedValsRed_1[i] = red_src[i];
    blendedValsRed_2[i] = red_src[i];
    blendedValsBlue_1[i] = blue_src[i];
    blendedValsBlue_2[i] = blue_src[i];
    blendedValsGreen_1[i] = green_src[i];
    blendedValsGreen_2[i] = green_src[i];
  }

  //Perform the solve on each color channel
  const size_t numIterations = 800;
  for (size_t i = 0; i < numIterations; ++i) {
    computeIteration(red_dst, strictInteriorPixels, borderPixels,
                     interiorPixelList, numColsSource, blendedValsRed_1, g_red,
                     blendedValsRed_2);

    std::swap(blendedValsRed_1, blendedValsRed_2);
  }

  for (size_t i = 0; i < numIterations; ++i) {
    computeIteration(blue_dst, strictInteriorPixels, borderPixels,
                     interiorPixelList, numColsSource, blendedValsBlue_1, g_blue,
                     blendedValsBlue_2);

    std::swap(blendedValsBlue_1, blendedValsBlue_2);
  }

  for (size_t i = 0; i < numIterations; ++i) {
    computeIteration(green_dst, strictInteriorPixels, borderPixels,
                     interiorPixelList, numColsSource, blendedValsGreen_1, g_green,
                     blendedValsGreen_2);

    std::swap(blendedValsGreen_1, blendedValsGreen_2);
  }
  std::swap(blendedValsRed_1,   blendedValsRed_2);   //put output into _2
  std::swap(blendedValsBlue_1,  blendedValsBlue_2);  //put output into _2
  std::swap(blendedValsGreen_1, blendedValsGreen_2); //put output into _2

  //copy the destination image to the output
  memcpy(h_blendedImg, h_destImg, sizeof(uchar4) * srcSize);

  //copy computed values for the interior into the output
  for (size_t i = 0; i < interiorPixelList.size(); ++i) {
    uint2 coord = interiorPixelList[i];

    unsigned int offset = coord.x * numColsSource + coord.y;

    h_blendedImg[offset].x = blendedValsRed_2[offset];
    h_blendedImg[offset].y = blendedValsBlue_2[offset];
    h_blendedImg[offset].z = blendedValsGreen_2[offset];
  }

  //wow, we allocated a lot of memory!
  delete[] mask;
  delete[] blendedValsRed_1;
  delete[] blendedValsRed_2;
  delete[] blendedValsBlue_1;
  delete[] blendedValsBlue_2;
  delete[] blendedValsGreen_1;
  delete[] blendedValsGreen_2;
  delete[] g_red;
  delete[] g_blue;
  delete[] g_green;
  delete[] red_src;
  delete[] red_dst;
  delete[] blue_src;
  delete[] blue_dst;
  delete[] green_src;
  delete[] green_dst;
  delete[] borderPixels;
  delete[] strictInteriorPixels;
}
#endif



  /* The reference calculation is provided below, feel free to use it
     for debugging purposes. 
   
    uchar4* h_reference = new uchar4[srcSize];
    reference_calc(h_sourceImg, numRowsSource, numColsSource,
                   h_destImg, h_reference);

    checkResultsEps((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * srcSize, 2, .01);
    delete[] h_reference;
  */ 

  checkCudaErrors(cudaFree(d_mask));
  checkCudaErrors(cudaFree(d_srcImg));

#if REF_JACOBI2
  delete [] g1;
  g1 = NULL;
  delete [] g2;
  g2 = NULL;
#endif
#if REF_JACOBI1
  delete [] g;
  g = NULL;
#endif
#if REF_BORI
  delete [] bori;
  bori = NULL;
#endif
#if REF_MASK
  delete [] mask;
  mask = NULL;
#endif

}

#endif // #if 0