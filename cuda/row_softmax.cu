// Educational CUDA counterpart. The Triton version is the executable reference.
#include <math.h>
extern "C" __global__ void row_softmax(const float* x, float* out, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) return;
  float maximum = -INFINITY;
  for (int j = 0; j < cols; ++j) maximum = fmaxf(maximum, x[row * cols + j]);
  float sum = 0.0f;
  for (int j = 0; j < cols; ++j) sum += expf(x[row * cols + j] - maximum);
  for (int j = 0; j < cols; ++j) out[row * cols + j] = expf(x[row * cols + j] - maximum) / sum;
}
