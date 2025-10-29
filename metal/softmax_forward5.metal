#include <metal_stdlib>
using namespace metal;
kernel void sum_forward5(
    device const float* A [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& denom [[buffer(3)]],
    constant float& global_max [[buffer(4)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        C[i]=exp(A[i]-global_max)/denom;
    }
}
