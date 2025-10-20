#include <metal_stdlib>
using namespace metal;
kernel void embedding_forward(
    device const float* embedding_matrix[[buffer(0)]],
    device const int* token_inds[[buffer(1)]],
    device float* output[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    constant uint& d[[buffer(4)]],
    uint i [[thread_position_in_grid]]
)
{
    if (i>=n) return;
    for (int j=0;j<d;j++){
        output[i*d+j]=embedding_matrix[token_inds[i]*d+j];
    }
}