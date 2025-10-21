#include <metal_stdlib>
using namespace metal;
kernel void cross_entropy_loss_backward(
    device const float* p[[buffer(0)]],
    device const float* q[[buffer(1)]],
    device const atomic_float* loss_grad[[buffer(2)]],
    device float* p_grad[[buffer(3)]],
    device float* q_grad[[buffer(4)]],
    constant uint& n[[buffer(5)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        float qi = clamp(q[i], 1e-9f, 1.0f);
        p_grad[i]=-loss_grad[0]*qi/n;
        q_grad[i]=-loss_grad[0]*p[i]/(n*qi);
    }
}
