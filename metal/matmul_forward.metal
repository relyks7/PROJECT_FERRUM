#include <metal_stdlib>
using namespace metal;
#define T 16
kernel void matmul_forward(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint2 j [[threadgroup_position_in_grid]]
)
{
    threadgroup float tA[T][T];
    threadgroup float tB[T][T];
    uint row=j.y*T+i.y;
    uint col=j.x*T+i.x;
    float acc=0;
    for (int curtile=0;curtile<(n+T-1)/T;curtile++){
        if (row < m && (curtile*T + i.x) < n)
            tA[i.x][i.y] = A[row*n + curtile*T + i.x];
        else
            tA[i.x][i.y] = 0.0f;

        if ((curtile*T + i.y) < n && col < p)
            tB[i.x][i.y] = B[(curtile*T + i.y)*p + col];
        else
            tB[i.x][i.y] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<T;idx++){
            acc+=tA[idx][i.y]*tB[i.x][idx];
        }
    }
    if (row<m&&col<p){
        C[row*p+col]=acc;
    }
}