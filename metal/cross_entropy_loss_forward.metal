#include <metal_stdlib>
using namespace metal;
kernel void cross_entropy_loss_forward(
    device const float* p[[buffer(0)]],
    device const float* q[[buffer(1)]],
    device atomic_float* loss[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        float qi = clamp(q[i], 1e-9f, 1.0f);
        atomic_fetch_add_explicit(&loss[0], p[i]*log(q[i])/(float)n, memory_order_relaxed);
    }
}
