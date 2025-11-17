#include <metal_stdlib>
using namespace metal;
#define T1 32
#define T2 8
#define K 32
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
    threadgroup float tA[T1][K];
    threadgroup float tB[K][T2];
    uint row=j.y*T1+i.y;
    uint col=j.x*T2+i.x;
    float acc=0;
    for (int curtile=0;curtile<(n+K-1)/K;curtile++){
        for (int idx=0;idx<K/T2;idx++){
            int idx_x=idx*T2+i.x;
            if (row < m && (curtile*K + idx_x) < n)
                tA[i.y][idx_x] = A[row*n + (curtile*K + idx_x)];
            else
                tA[i.y][idx_x] = 0.0f;
        }
        if ((curtile*K + i.y) < n && col < p)
            tB[i.y][i.x] = B[(curtile*K + i.y)*p + col];
        else
            tB[i.y][i.x] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<K;idx++){
            acc+=tA[i.y][idx]*tB[idx][i.x];
        }
    }
    if (row<m&&col<p){
        C[row*p+col]+=acc;
    }
}