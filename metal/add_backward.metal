#include <metal_stdlib>
using namespace metal;
kernel void add_backward(
    device float* A_grad[[buffer(0)]],
    device float* B_grad[[buffer(1)]],
    device const float* C_grad[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        A_grad[i]+=C_grad[i];
        B_grad[i]+=C_grad[i];
    }
}
