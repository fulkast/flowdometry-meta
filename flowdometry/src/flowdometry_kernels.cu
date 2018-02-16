#include <flowdometry_kernels.h>
#include <utils.h>

#include <vector>
#include <stdio.h>

int divUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

__global__ void shmem_reduce_float_with_operation(float * d_out, const float * d_in, const int is_min)
{

    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = isnan(d_in[myId]) ? 0 : d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (is_min ) {
                sdata[tid] = fmin(sdata[tid + s], sdata[tid]);
            } else {
                sdata[tid] = fmax(sdata[tid + s], sdata[tid]);
                // std::printf("blockdimx %i ", (int)blockDim.x);
            }
        }
        __syncthreads();        // make sure all comparisons at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}


__global__ void shmem_reduce_normal_eqs_flow_GPU(float * d_A_intermediate, const float* const d_x_flow,
                                    const float* const d_y_flow,
                                    const float* const d_depth,
                                    float fx, float fy, float ox, float oy,
                                    int blocks,
                                    int threads)
{

    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float d_A_sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    for (int i = 0; i < 23; i++) {
      d_A_sdata[tid*23 + i] = 0;
    }

    // load shared mem from global mem
    float disp = d_depth[myId];
    float ux = d_x_flow[myId];
    float uy = d_y_flow[myId];

    if (!isnan(disp) && !isnan(ux) && !isnan(uy) && disp > 0.0001 && disp < 3.0) {

      // compute coordinates
      int pixel_ind = myId;

      float y = floorf(__fdividef((float)pixel_ind, threads));
      float x = (float)pixel_ind - y * threads;

      x = x - ox;
      y = y - oy;

      // unique values A-matrix
      d_A_sdata[tid*23 + 0] += (disp * disp * fx * fx);
      d_A_sdata[tid*23 + 1] += (-disp * disp * x * fx);
      d_A_sdata[tid*23 + 2] += (-disp * x * y);
      d_A_sdata[tid*23 + 3] += (disp * fx * fx + disp * x * x);
      d_A_sdata[tid*23 + 4] += (-disp * y * fx);
      d_A_sdata[tid*23 + 5] += (-disp * disp * y * fy);
      d_A_sdata[tid*23 + 6] += (-disp * fy * fy - disp * y * y); //!!!!
      d_A_sdata[tid*23 + 7] += (disp * x * fy);
      d_A_sdata[tid*23 + 8] += (disp * disp * x * x + disp * disp * y * y);
      d_A_sdata[tid*23 + 9] += (disp * x * x * y / fx + disp * y * fy + disp * y * y * y / fy);
      d_A_sdata[tid*23 + 10] += (-disp * x * fx - disp * x * x * x / fx - disp * x * y * y / fy);
      d_A_sdata[tid*23 + 11] += (x * x * y * y / (fx * fx) + fy * fy + 2.0f * y * y +
              y * y * y * y / (fy * fy));
      d_A_sdata[tid*23 + 12] += (-2.0f * x * y - x * x * x * y / (fx * fx) -
              x * y * y * y / (fy * fy));
      d_A_sdata[tid*23 + 13] += (x * y * y / fx - x * fy - x * y * y / fy);
      d_A_sdata[tid*23 + 14] += (fx * fx + 2.0f * x * x + x * x * x * x / (fx * fx) +
              x * x * y * y / (fy * fy));
      d_A_sdata[tid*23 + 15] += (-y * fx - x * x * y / fx + x * x * y / fy);
      d_A_sdata[tid*23 + 16] += (x * x + y * y);

      // B-vector

      d_A_sdata[tid*23 + 17] += (disp * ux * fx);
      d_A_sdata[tid*23 + 18] += (disp * uy * fy);
      d_A_sdata[tid*23 + 19] += (-disp * x * ux - disp * y * uy);
      d_A_sdata[tid*23 + 20] += (-x * y * ux / fx - uy * fy - uy * y * y / fy);
      d_A_sdata[tid*23 + 21] += (ux * fx + x * x * ux / fx + x * y * uy / fy);
      d_A_sdata[tid*23 + 22] += (-y * ux + x * uy);

    }

    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
          for (int i = 0; i < 23; i++) {
            d_A_sdata[tid*23 + i] += d_A_sdata[(tid + s)*23 + i];
          }
        }
        __syncthreads();        // make sure all comparisons at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        for (int i = 0; i < 23; i++) {
          d_A_intermediate[blockIdx.x*23+i] = d_A_sdata[0*23+i];
        }
    }
}

__global__ void shmem_reduce_vector_GPU(float * d_A_intermediate, const float * d_in,
                                    int blocks,
                                    int threads)
{

    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float d_A_sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    for (int i = 0; i < 23; i++) {
      d_A_sdata[tid*23 + i] = d_in[myId * 23 + i];
    }

    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
          for (int i = 0; i < 23; i++) {
            d_A_sdata[tid*23 + i] += d_A_sdata[(tid + s)*23 + i];
          }
        }
        __syncthreads();        // make sure all comparisons at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        for (int i = 0; i < 23; i++) {
          d_A_intermediate[blockIdx.x*23+i] = d_A_sdata[0*23+i];
        }
    }
}

void normal_eqs_flow_GPU(float* A, float* d_A_intermediate, const float* const d_x_flow,
                                    const float* const d_y_flow,
                                    const float* const d_depth,
                                    float fx, float fy, float ox, float oy,
                                    int blocks,
                                    int threads)

{

  struct cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, 0);

  size_t block_shared_mem = threads * 23 * sizeof(float);

  assert(block_shared_mem < prop.sharedMemPerBlock &&
      "Attempting to allocate more block shared memory than available. \
       Try downscale the input image more" );

  // find the intermediate minimum
  shmem_reduce_normal_eqs_flow_GPU
  <<<blocks, threads, block_shared_mem>>>
  (d_A_intermediate, d_x_flow, d_y_flow, d_depth, fx, fy, ox, oy, blocks,threads);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
      fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",
               __FILE__, __LINE__, cudaGetErrorString(err) );
      exit(EXIT_FAILURE);
  }

  // reduce to the final value
  threads = blocks;
  blocks = 1;
  float * d_output;

  checkCudaErrors( cudaMalloc(&d_output, sizeof(float)*23) );

  shmem_reduce_vector_GPU
  <<<blocks, threads, threads * 23 * sizeof(float)>>>
  (d_output, d_A_intermediate, blocks, threads);

  checkCudaErrors( cudaMemcpy(A, d_output, sizeof(float) * 23, cudaMemcpyDeviceToHost) );

  // free locally declared variables
  checkCudaErrors( cudaFree(d_output) );

}
