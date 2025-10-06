#include <metal_stdlib>
using namespace metal;
kernel void copy_backward(
    device float* A_grad[[buffer(0)]],
    device const float* C_grad[[buffer(1)]],
    constant uint& n[[buffer(2)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        A_grad[i]+=C_grad[i];
    }
}
