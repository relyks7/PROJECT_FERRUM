#include <metal_stdlib>
using namespace metal;
#define T 16
kernel void hebbian_oja_backward8(
    device const float* Z_grad [[buffer(0)]],
    device const float* U [[buffer(1)]],
    device float* V_grad [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint2 j [[threadgroup_position_in_grid]]
)
{
    threadgroup float tZ_grad_t[T][T];
    threadgroup float tU[T][T];
    uint row=j.y*T+i.y;
    uint col=j.x*T+i.x;
    float acc=0;
    for (int curtile=0;curtile<(m+T-1)/T;curtile++){
        if ((curtile*T + i.x) < m && row < n)
            tZ_grad_t[i.x][i.y] = Z_grad[(curtile*T+i.x)*n+row];
        else
            tZ_grad_t[i.x][i.y] = 0.0f;

        if (col < p && (curtile*T+i.x)<m)
            tU[i.x][i.y] = U[(curtile*T + i.x)*p+col];
        else
            tU[i.x][i.y] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<T;idx++){
            acc+=tZ_grad_t[idx][i.y]*tU[idx][i.x];
        }
    }
    if (row<n&&col<p){
        V_grad[row*p+col]+=acc;
    }
}