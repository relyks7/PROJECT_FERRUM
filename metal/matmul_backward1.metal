#include <metal_stdlib>
using namespace metal;
#define T 16
kernel void matmul_backward1(
    device const float* B [[buffer(0)]],
    device const float* C_grad [[buffer(1)]],
    device float* A_grad [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint2 j [[threadgroup_position_in_grid]]
)
{
    threadgroup float tC_grad[T][T];
    threadgroup float tB_t[T][T];
    uint row=j.y*T+i.y;
    uint col=j.x*T+i.x;
    float acc=0;
    for (int curtile=0;curtile<(p+T-1)/T;curtile++){
        if (row < m && (curtile*T + i.x) < p)
            tC_grad[i.x][i.y] = C_grad[row*p + curtile*T + i.x];
        else
            tC_grad[i.x][i.y] = 0.0f;

        if ((curtile*T + i.y) < p && col < n)
            tB_t[i.x][i.y] = B[col * p + (curtile*T + i.y)];
        else
            tB_t[i.x][i.y] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<T;idx++){
            acc+=tC_grad[idx][i.y]*tB_t[i.x][idx];
        }
    }
    if (row<m&&col<n){
        A_grad[row*n+col]=acc;
    }
}