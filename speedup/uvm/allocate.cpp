// Custom GPU memory allocator used by the UVM baseline in speedup/uvm/.
//
// PyTorch's default allocator calls cudaMalloc(), which allocates strictly
// GPU-resident memory — when VRAM is exhausted, allocations fail. This
// directory implements an alternative offloading strategy that runs OPT
// inference larger than GPU memory by letting CUDA's Unified Memory
// subsystem page-migrate between CPU RAM and GPU VRAM automatically.
//
// To make that work, every tensor PyTorch creates must come from
// cudaMallocManaged() instead of cudaMalloc(). PyTorch does not expose
// cudaMallocManaged() directly, so this file exposes the two C-ABI symbols
// (uvm_malloc / uvm_free) that PyTorch's CUDAPluggableAllocator expects.
//
// Build into allocate.so and load from Python via:
//     torch.cuda.memory.CUDAPluggableAllocator(
//         'allocate.so', 'uvm_malloc', 'uvm_free')
// see transformer.py:71-73.
//
// extern "C" is required because the pluggable-allocator interface looks
// up the symbols by their unmangled C names via dlopen/dlsym.
#include <sys/types.h>
#include <cuda_runtime_api.h>

extern "C" {
  // Called by PyTorch every time a CUDA tensor needs backing memory.
  // Returning a pointer from cudaMallocManaged() makes the allocation
  // addressable from both CPU and GPU; the CUDA driver migrates pages
  // on demand when a kernel or host code touches them. This is what lets
  // KV caches and weights that exceed VRAM still "fit" — pages spill to
  // CPU and are faulted back in on access.
  //
  // `device` and `stream` are part of the required signature but unused:
  // managed allocations are not bound to a specific device or stream.
  void* uvm_malloc(ssize_t size, int device, cudaStream_t stream) {
    void *ptr;
    //cudaMalloc(&ptr, size);   // ← the default behavior we are replacing
    cudaMallocManaged(&ptr, size);
    return ptr;
  }

  // Counterpart used when PyTorch reclaims a tensor's storage. cudaFree
  // works for both regular and managed allocations, so no special path
  // is needed here.
  void uvm_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    cudaFree(ptr);
  }
}
