#include <metal_stdlib>
using namespace metal;
kernel void mul_backward(
    device float* A[[buffer(0)]],
    device float* B[[buffer(1)]],
    device float* A_grad[[buffer(2)]],
    device float* B_grad[[buffer(3)]],
    device const float* C_grad[[buffer(4)]],
    constant uint& n[[buffer(5)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        A_grad[i]+=C_grad[i]*B[i];
        B_grad[i]+=C_grad[i]*A[i];
    }
}
