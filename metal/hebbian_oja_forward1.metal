#include <metal_stdlib>
using namespace metal;
#define T 16
kernel void hebbian_oja_forward1(
    device const float* V [[buffer(0)]],
    device const float* U [[buffer(1)]],
    device float* eta [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    constant float& etanull [[buffer(6)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint2 j [[threadgroup_position_in_grid]]
)
{
    threadgroup float tU[T][T];
    threadgroup float tV_t[T][T];
    uint row=j.y*T+i.y;
    uint col=j.x*T+i.x;
    float acc=0;
    for (int curtile=0;curtile<(p+T-1)/T;curtile++){
        if (row < m && (curtile*T + i.x) < p)
            tU[i.x][i.y] = U[row*p + curtile*T + i.x];
        else
            tU[i.x][i.y] = 0.0f;

        if ((curtile*T + i.y) < p && col < n)
            tV_t[i.x][i.y] = V[col * p + (curtile*T + i.y)];
        else
            tV_t[i.x][i.y] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<T;idx++){
            acc+=tU[idx][i.y]*tV_t[i.x][idx];
        }
    }
    if (row<m&&col<n){
        eta[row*n+col]=etanull*tanh(acc);
    }
}