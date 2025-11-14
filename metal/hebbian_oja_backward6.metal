#include <metal_stdlib>
using namespace metal;
kernel void hebbian_oja_backward6(
    device const float* eta_grad[[buffer(0)]],
    device const float* eta[[buffer(1)]],
    device float* Z_grad[[buffer(2)]],
    constant float& etanull[[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint i[[thread_position_in_grid]]
) {
    if (i<n){
        Z_grad[i]+=eta_grad[i]*(1-(eta[i]/etanull)*(eta[i]/etanull));
    }
}
