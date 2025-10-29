#include <metal_stdlib>
using namespace metal;
kernel void softmax_forward2(
    device const float* reduced_max [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]],
) {
    if (i<n){
        atomic_fetch_max_explicit(&C[0], reduced_max[i], memory_order_relaxed);
    }
}
