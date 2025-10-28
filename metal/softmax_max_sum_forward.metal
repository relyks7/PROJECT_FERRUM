#include <metal_stdlib>
using namespace metal;
#define T 128
#define WARPS 16
kernel void softmax_forward(
    device const float* A [[buffer(0)]],
    device float* max_out [[buffer(1)]],
    device float* sum_out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_threadgroup]],
    uint j [[thread_position_in_grid]],
    uint si [[thread_index_in_simdgroup]],
    uint sj [[simdgroup_index_in_threadgroup]]
) {
    if (j>=n){
        return;
    }
    float val=A[j];
    float local_max=simdgroup_reduce_max(val);
    float 
    threadgroup float pm[WARPS];
    if (simd_is_first()){
        pm[sj]=local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sj==0){
        float x = (si < WARPS) ? pm[si] : -INFINITY;
        float final_max=simdgroup_reduce_max(x);
        if (simd_is_first()){
            max_out[i]=final_max;
        }
    }
}
