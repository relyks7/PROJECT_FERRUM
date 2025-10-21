#include <metal_stdlib>
using namespace metal;
kernel void embedding_backward(
    device atomic_float* embedding_grad[[buffer(0)]],
    device const int* token_inds[[buffer(1)]],
    device const float* token_grads[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    constant uint& d[[buffer(4)]],
    uint i [[thread_position_in_grid]]
)
{
    if (i>=n) return;
    for (int j=0;j<d;j++){
        atomic_fetch_add_explicit(&embedding_grad[token_inds[i] * d + j],
                                  token_grads[i * d + j],
                                  memory_order_relaxed);
    }
}