#include <metal_stdlib>
using namespace metal;
#define T 16
kernel void matmul_backward(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* A_grad [[buffer(2)]],
    device float* B_grad [[buffer(3)]],
    device float* C_grad [[buffer(4)]],
    constant uint& m [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& p [[buffer(7)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint2 j [[threadgroup_position_in_grid]]
)
{
    threadgroup float tC_grad[T][T];
    threadgroup float tB_t[T][T];
    threadgroup float tA_t[T][T];
    uint row=j.y*T+i.y;
    uint col=j.x*T+i.x;
    float acc=0;
    for (int curtile=0;curtile<(p+T-1)/T;curtile++){
        if (row < m && (curtile*T + i.x) < p)
            tC_grad[i.y][i.x] = C_grad[row*p + curtile*T + i.x];
        else
            tC_grad[i.y][i.x] = 0.0f;

        if ((curtile*T + i.y) < p && col < n)
            tB_t[i.y][i.x] = B[col*p + (curtile*T + i.y)];
        else
            tB_t[i.y][i.x] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<T;idx++){
            acc+=tC_grad[i.y][idx]*tB_t[idx][i.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row<m&&col<n){
        A_grad[row*n+col]=acc;
    }
    acc=0;
    for (int curtile=0;curtile<(m+T-1)/T;curtile++){
        if ((curtile*T + i.x) < m && row < n)
            tA_t[i.y][i.x] = A[(curtile*T + i.x) * n + row];
        else
            tA_t[i.y][i.x] = 0.0f;
        if ((curtile*T + i.y) < m && col < p)
            tC_grad[i.y][i.x] = C_grad[(curtile*T + i.y) * p + col];
        else
            tC_grad[i.y][i.x] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx=0;idx<T;idx++){
            acc+=tC_grad[i.y][idx]*tA_t[idx][i.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row<n&&col<p){
        B_grad[row*p+col]=acc;
    }
}