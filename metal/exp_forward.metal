#include <metal_stdlib>
using namespace metal;
kernel void exp_forward(
    device const float* A[[buffer(0)]],
    device float* C[[buffer(1)]],
    constant uint& n[[buffer(2)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        C[i]=exp(A[i]);
    }
}
