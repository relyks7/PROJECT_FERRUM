#include <metal_stdlib>
using namespace metal;
#define T 16
kernel void hebbian_oja_forward2(
    device const float* X [[buffer(0)]],
    device const float* Y [[buffer(1)]],
    device float* otpt [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint2 j [[threadgroup_position_in_grid]]
)
{
    threadgroup float tY[T][T];
    threadgroup float tX_t[T][T];
    uint row=j.y*T+i.y;
    uint col=j.x*T+i.x;
    float acc=0;
    for (int curtile=0;curtile<(p+T-1)/T;curtile++){
        if (row < m && (curtile*T + i.x) < p)
            tY[i.x][i.y] = Y[row*p + curtile*T + i.x];
        else
            tY[i.x][i.y] = 0.0f;

        if ((curtile*T + i.y) < p && col < n)
            tX_t[i.x][i.y] = X[col * p + (curtile*T + i.y)];
        else
            tX_t[i.x][i.y] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<T;idx++){
            acc+=tY[idx][i.y]*tX_t[i.x][idx];
        }
    }
    if (row<m&&col<n){
        otpt[row*n+col]=acc;
    }
}