#include <metal_stdlib>
using namespace metal;
#define T 16
kernel void hebbian_oja_backward3(
    device const float* M [[buffer(0)]],
    device const float* N_grad [[buffer(1)]],
    device float* W_grad [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint2 j [[threadgroup_position_in_grid]]
)
{
    threadgroup float tM_t[T][T];
    threadgroup float tN_grad[T][T];
    uint row=j.y*T+i.y;
    uint col=j.x*T+i.x;
    float acc=0;
    for (int curtile=0;curtile<(m+T-1)/T;curtile++){
        if ((curtile*T + i.x) < m && row < n)
            tM_t[i.x][i.y] = M[(curtile*T+i.x)*n+row];
        else
            tM_t[i.x][i.y] = 0.0f;

        if (col < p && (curtile*T+i.x)<m)
            tN_grad[i.x][i.y] = N_grad[(curtile*T + i.x)*p+col];
        else
            tN_grad[i.x][i.y] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<T;idx++){
            acc+=tM_t[idx][i.y]*tN_grad[idx][i.x];
        }
    }
    if (row<n&&col<p){
        W_grad[row*p+col]+=acc;
    }
}