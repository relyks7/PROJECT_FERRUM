#include <metal_stdlib>
using namespace metal;
kernel void softmax_forward(
    device const float* A[[buffer(0)]],
    device atomic_float* C[[buffer(1)]],
    constant uint& n[[buffer(2)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        atomic_fetch_add_explicit(&C[0], A[i], memory_order_relaxed);
    }
}
