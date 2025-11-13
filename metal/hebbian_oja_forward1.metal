#include <metal_stdlib>
using namespace metal;
#define T 16
kernel void hebbian_oja_forward1(
    device const float* U [[buffer(0)]],
    device float* eta [[buffer(1)]],
    device const float* V [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    constant uint& etalnull [[buffer(6)]],
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
            tU[i.y][i.x] = U[row*p + curtile*T + i.x];
        else
            tU[i.y][i.x] = 0.0f;

        if ((curtile*T + i.y) < p && col < n)
            tV_t[i.y][i.x] = V[col*p + (curtile*T + i.y)];
        else
            tV_t[i.y][i.x] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<T;idx++){
            acc+=tU[i.y][idx]*tV_t[idx][i.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row<m&&col<n){
        eta[row*n+col]=etalnull*tanh(acc);
    }
}