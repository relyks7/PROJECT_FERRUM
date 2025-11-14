#include <metal_stdlib>
using namespace metal;
#define T 16
kernel void hebbian_oja_forward3(
    device const float* D [[buffer(0)]],
    device const float* E [[buffer(1)]],
    device float* W [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint2 j [[threadgroup_position_in_grid]]
)
{
    threadgroup float tE[T][T];
    threadgroup float tD_t[T][T];
    uint row=j.y*T+i.y;
    uint col=j.x*T+i.x;
    float acc=0;
    for (int curtile=0;curtile<(p+T-1)/T;curtile++){
        if (row < m && (curtile*T + i.x) < p)
            tE[i.x][i.y] = E[row*p + curtile*T + i.x];
        else
            tE[i.x][i.y] = 0.0f;

        if ((curtile*T + i.y) < p && col < n)
            tD_t[i.x][i.y] = D[col * p + (curtile*T + i.y)];
        else
            tD_t[i.x][i.y] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<T;idx++){
            acc+=tE[idx][i.y]*tD_t[i.x][idx];
        }
    }
    if (row<m&&col<n){
        W[row*n+col]=acc;
    }
}