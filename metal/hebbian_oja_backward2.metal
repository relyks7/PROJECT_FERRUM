#include <metal_stdlib>
using namespace metal;
#define T 16
kernel void hebbian_oja_backward2(
    device const float* W [[buffer(0)]],
    device const float* N_grad [[buffer(1)]],
    device const float* Y [[buffer(2)]],
    device float* Y_grad [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& p [[buffer(6)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint2 j [[threadgroup_position_in_grid]]
)
{
    threadgroup float tN_grad[T][T];
    threadgroup float tW_t[T][T];
    uint row=j.y*T+i.y;
    uint col=j.x*T+i.x;
    float acc=0;
    for (int curtile=0;curtile<(p+T-1)/T;curtile++){
        if (row < m && (curtile*T + i.x) < p)
            tN_grad[i.x][i.y] = N_grad[row*p + curtile*T + i.x];
        else
            tN_grad[i.x][i.y] = 0.0f;

        if ((curtile*T + i.y) < p && col < n)
            tW_t[i.x][i.y] = W[col * p + (curtile*T + i.y)];
        else
            tW_t[i.x][i.y] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<T;idx++){
            acc+=tN_grad[idx][i.y]*tW_t[i.x][idx];
        }
    }
    if (row<m&&col<n){
        Y_grad[row*n+col]+=acc*2*Y[row*n+col];
    }
}