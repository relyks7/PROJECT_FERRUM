#include <metal_stdlib>
using namespace metal;
kernel void mask_backward(
    device float* B[[buffer(0)]],
    device float* A_grad[[buffer(1)]],
    device const float* C_grad[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        A_grad[i]+= (B[i]==0) ? 0:C_grad[i];
    }
}
