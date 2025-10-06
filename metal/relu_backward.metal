#include <metal_stdlib>
using namespace metal;
kernel void relu_backward(
    device const float* A[[buffer(0)]],
    device float* A_grad[[buffer(1)]],
    device const float* C_grad[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        A_grad[i]+=(A[i]>0)? C_grad[i]:0;
    }
}
