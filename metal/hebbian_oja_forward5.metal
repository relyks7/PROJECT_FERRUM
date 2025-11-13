#include <metal_stdlib>
using namespace metal;
kernel void hebbian_oja_forward4(
    device const float* eta[[buffer(0)]],
    device const float* heb[[buffer(1)]],
    device const float* yw[[buffer(2)]],
    device float* d[[buffer(3)]],
    constant uint& n[[buffer(4)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        d[i]=eta[i]*(heb[i]-yw[i]);
    }
}
