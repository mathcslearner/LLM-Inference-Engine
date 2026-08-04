#include "common/cuda.h"

#include "cuda/cuda_utils.h"

namespace engine::testing {

bool HasCudaDevice() { return cuda::device_count() > 0; }

}  // namespace engine::testing
