#include <metal_stdlib>
using namespace metal;
kernel void hebbian_oja_backward1(
    device const float* C_grad[[buffer(0)]],
    device const float* H[[buffer(1)]],
    device const float* N[[buffer(2)]],
    device const float* eta[[buffer(3)]],
    device float* H_grad[[buffer(4)]],
    device float* N_grad[[buffer(5)]],
    device float* eta_grad[[buffer(6)]],
    constant uint& n[[buffer(7)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        eta_grad[i]+=C_grad[i]*(H[i]-N[i]);
        H_grad[i]+=C_grad[i]*eta[i];
        N_grad[i]+=-C_grad[i]*eta[i];
    }
}
