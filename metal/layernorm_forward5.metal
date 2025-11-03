#include <metal_stdlib>
using namespace metal;
kernel void layernorm_forward5(
    device const float* mu [[buffer(0)]],
    device const float* sigma2 [[buffer(1)]],
    device const float* A [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    device const float* gamma [[buffer(5)]],
    device const float* beta [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        C[i]=gamma[i]* (A[i]-mu[0])/sqrt(sigma2[0]+eps) + beta[i];
    }
}
