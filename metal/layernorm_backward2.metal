#include <metal_stdlib>
using namespace metal;
#define T 128
#define WARPS 16
kernel void layernorm_backward2(
    device const float* A1 [[buffer(0)]],
    device float* sum_out_1 [[buffer(1)]],
    device const float* A2 [[buffer(2)]],
    device float* sum_out_2 [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint i [[thread_position_in_threadgroup]],
    uint j [[thread_position_in_grid]],
    uint k [[threadgroup_position_in_grid]],
    uint si [[thread_index_in_simdgroup]],
    uint sj [[simdgroup_index_in_threadgroup]]
) {
    if (j>=n){
        return;
    }
    float val1=A1[j];
    float local_sum1=simdgroup_reduce_sum(val1);
    float val2=A2[j];
    float local_sum2=simdgroup_reduce_sum(val2);
    threadgroup float ps1[WARPS];
    threadgroup float ps2[WARPS];
    if (simd_is_first()){
        ps1[sj]=local_sum1;
        ps2[sj]=local_sum2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sj==0){
        float xs1 = (si < WARPS) ? ps1[si] : 0.0f;
        float final_sum1=simdgroup_reduce_sum(xs1);
        float xs2 = (si < WARPS) ? ps2[si] : 0.0f;
        float final_sum2=simdgroup_reduce_sum(xs2);
        if (simd_is_first() && i==0){
            sum_out_1[k]=final_sum1;
            sum_out_2[k]=final_sum2;
        }
    }
}
