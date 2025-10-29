#include <metal_stdlib>
using namespace metal;
kernel void max_backward(
    device float* A_grad[[buffer(0)]],
    device const float* C_grad[[buffer(1)]],
    device const float* A [[buffer(2)]],
    device const float* C [[buffer(3)]],
    constant uint& n[[buffer(4)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        if (fabs(A[i]-C[0])<1e-6){
            A_grad[i]+=C_grad[0];
        }
    }
}
