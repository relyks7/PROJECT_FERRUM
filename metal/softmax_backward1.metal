#include <metal_stdlib>
using namespace metal;
#define T 128
#define WARPS 16
kernel void softmax_backward1(
    device const float* C [[buffer(0)]],
    device const float* C_grad [[buffer(1)]],
    device float* sum_out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_threadgroup]],
    uint j [[thread_position_in_grid]],
    uint k [[threadgroup_position_in_grid]],
    uint si [[thread_index_in_simdgroup]],
    uint sj [[simdgroup_index_in_threadgroup]]
) {
    if (j>=n){
        return;
    }
    float val=C[j]*C_grad[j];
    float local_sum=simdgroup_reduce_sum(val);
    threadgroup float ps[WARPS];
    if (simd_is_first()){
        ps[sj]=local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sj==0){
        float xs = (si < WARPS) ? ps[si] : 0.0f;
        float final_sum=simdgroup_reduce_sum(xs);
        if (simd_is_first() && i==0){
            sum_out[k]=final_sum;
        }
    }
}
