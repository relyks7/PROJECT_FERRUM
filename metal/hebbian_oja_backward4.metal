#include <metal_stdlib>
using namespace metal;
#define T 16
kernel void hebbian_oja_backward4(
    device const float* H_grad [[buffer(0)]],
    device const float* Y [[buffer(1)]],
    device float* X_grad [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint2 j [[threadgroup_position_in_grid]]
)
{
    threadgroup float tH_grad[T][T];
    threadgroup float tY[T][T];
    uint row=j.y*T+i.y;
    uint col=j.x*T+i.x;
    float acc=0;
    for (int curtile=0;curtile<(n+T-1)/T;curtile++){
        if (row < m && (curtile*T + i.x) < n)
            tH_grad[i.x][i.y] = H_grad[row*n + curtile*T + i.x];
        else
            tH_grad[i.x][i.y] = 0.0f;

        if ((curtile*T + i.y) < n && col < p)
            tY[i.x][i.y] = Y[(curtile*T + i.y)*p + col];
        else
            tY[i.x][i.y] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<T;idx++){
            acc+=tH_grad[idx][i.y]*tY[i.x][idx];
        }
    }
    if (row<m&&col<p){
        X_grad[row*p+col]+=acc;
    }
}