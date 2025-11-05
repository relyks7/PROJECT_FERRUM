#include <metal_stdlib>
using namespace metal;
kernel void layernorm_backward1(
    device const float* sigma2 [[buffer(0)]],
    device const float* A [[buffer(1)]],
    device const float* C [[buffer(2)]],
    device const float* C_grad [[buffer(3)]],
    device float* A_grad [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    device const float* gamma [[buffer(6)]],
    device float* gamma_grad [[buffer(7)]],
    device float* beta_grad [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    device const float* mu [[buffer(10)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        beta_grad[i]+=C_grad[i];
        gamma_grad[i]+=C_grad[i]*(A[i]-mu[0])/sqrt(sigma2[0]+eps);
        frac_grad=C_grad[i]*gamma[i]
    }
}
