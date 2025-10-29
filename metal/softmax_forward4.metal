#include <metal_stdlib>
using namespace metal;
kernel void sum_forward4(
    device const float* reduced_sum [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        atomic_fetch_add_explicit(&C[0], reduced_sum[i], memory_order_relaxed);
    }
}
