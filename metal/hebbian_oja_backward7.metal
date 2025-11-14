#include <metal_stdlib>
using namespace metal;
#define T 16
kernel void hebbian_oja_backward7(
    device const float* Z_grad [[buffer(0)]],
    device const float* V [[buffer(1)]],
    device float* U_grad [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint2 j [[threadgroup_position_in_grid]]
)
{
    threadgroup float tZ_grad[T][T];
    threadgroup float tV[T][T];
    uint row=j.y*T+i.y;
    uint col=j.x*T+i.x;
    float acc=0;
    for (int curtile=0;curtile<(n+T-1)/T;curtile++){
        if (row < m && (curtile*T + i.x) < n)
            tZ_grad[i.x][i.y] = Z_grad[row*n + curtile*T + i.x];
        else
            tZ_grad[i.x][i.y] = 0.0f;

        if ((curtile*T + i.y) < n && col < p)
            tV[i.x][i.y] = V[(curtile*T + i.y)*p + col];
        else
            tV[i.x][i.y] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<T;idx++){
            acc+=tZ_grad[idx][i.y]*tV[i.x][idx];
        }
    }
    if (row<m&&col<p){
        U_grad[row*p+col]+=acc;
    }
}