#include <metal_stdlib>
using namespace metal;
#define T 16
kernel void hebbian_oja_forward3(
    device const float* Y  [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint2 j [[threadgroup_position_in_grid]]
)
{
    threadgroup float tY[T][T];
    threadgroup float tW[T][T];
    uint row=j.y*T+i.y;
    uint col=j.x*T+i.x;
    float acc=0;
    for (int curtile=0;curtile<(n+T-1)/T;curtile++){
        if (row < m && (curtile*T + i.x) < n)
            tY[i.y][i.x] = Y[row*n + curtile*T + i.x]*Y[row*n + curtile*T + i.x];
        else
            tY[i.y][i.x] = 0.0f;

        if ((curtile*T + i.y) < n && col < p)
            tW[i.y][i.x] = W[(curtile*T + i.y)*p + col];
        else
            tW[i.y][i.x] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<T;idx++){
            acc+=tY[i.y][idx]*tW[idx][i.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row<m&&col<p){
        C[row*p+col]=acc;
    }
}