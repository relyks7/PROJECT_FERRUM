#include <metal_stdlib>
using namespace metal;
kernel void layernorm_backward3(
    device const float* reduced_sum1 [[buffer(0)]],
    device const float* reduced_sum2 [[buffer(1)]],
    device const float* mu [[buffer(2)]],
    device const float* sigma2 [[buffer(3)]],
    device const float* A [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    device const float* gamma [[buffer(6)]],
    device const float* beta [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    device float* A_grad [[buffer(9)]],
    device float* C_grad [[buffer(10)]],
    device float* gamma_grad [[buffer(11)]],
    device float* beta_grad [[buffer(12)]],
    device const float* final_sum1 [[buffer(13)]],
    device const float* final_sum2 [[buffer(14)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        atomic_fetch_add_explicit(&final_sum1[0], reduced_sum1[i], memory_order_relaxed);
        atomic_fetch_add_explicit(&final_sum2[0], reduced_sum2[i], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (i<n){
        beta_grad[i]+=C_grad[i];
        gamma_grad[i]+=(A[i]-mu[0])*(C_grad[i])/(sigma2[0]+eps);
        A_grad[i]+=sqrt(sigma2[0]+eps)*(gamma[i]*C_grad[i] - final_sum1[0]/n - (A[i]-mu[0])*final_sum2[0]/(n*(sigma2[0]+eps)));
    }
}
