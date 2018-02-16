#include <cuda_runtime.h>

int divUp(int a, int b);

void normal_eqs_flow_GPU(float* A, float* d_A_intermediate, const float* const d_x_flow,
                                    const float* const d_y_flow,
                                    const float* const d_depth,
                                    float fx, float fy, float ox, float oy,
                                    int blocks,
                                    int threads);
