#include <metal_stdlib>
using namespace metal;
kernel void softmax_backward3(
    device const float* reduced_sum [[buffer(0)]],
    device const float* C [[buffer(1)]],
    device const float* C_grad [[buffer(2)]],
    device float* A_grad [[buffer(3)]],
    device float* s [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        A_grad[i]=C[i]*(C_grad[i]-s[0]);
    }
}
