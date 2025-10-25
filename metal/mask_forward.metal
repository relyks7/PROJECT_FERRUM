#include <metal_stdlib>
using namespace metal;
kernel void mask_forward(
    device const float* A[[buffer(0)]],
    device const float* B[[buffer(1)]],
    device float* C[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        C[i]= (B[i]==0) ? 0:A[i];
    }
}