#include <metal_stdlib>
using namespace metal;
#define T 16
kernel void matmul_backward2(
    device const float* A [[buffer(0)]],
    device float* B_grad [[buffer(1)]],
    device float* C_grad [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint2 j [[threadgroup_position_in_grid]]
)
{
    threadgroup float tC_grad[T][T];
    threadgroup float tA_t[T][T];
    uint row=j.y*T+i.y;
    uint col=j.x*T+i.x;
    float acc=0;
    acc=0;
    for (int curtile=0;curtile<(m+T-1)/T;curtile++){
        if ((curtile*T + i.x) < m && row < n)
            tA_t[i.y][i.x] = A[row*n + (curtile*T + i.x)];
        else
            tA_t[i.y][i.x] = 0.0f;
        if ((curtile*T + i.y) < m && col < p)
            tC_grad[i.y][i.x] = C_grad[(curtile*T + i.y) * p + col];
        else
            tC_grad[i.y][i.x] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<T;idx++){
            acc+=tC_grad[idx][i.x]*tA_t[idx][i.y];
        }
    }
    if (row<n&&col<p){
        B_grad[row*p+col]=acc;
    }
}