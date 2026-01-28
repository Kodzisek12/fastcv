#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <cmath>
#include <nvtx3/nvToolsExt.h>
#include "utils.cuh"   
torch::Tensor GaussianBlur(torch::Tensor img, int blurSize){
    std::cout << "Gaussian Blur CUDA kernel called with blur size: " << blurSize << std::endl;
    return img;
}

__global__ void GaussianBlurKernel(
    unsigned char* src, 
    unsigned char* dst,
    int width, int height,
    int channels,
    int blursize, 
    double sigmaX,
    double sigmaY)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int radius = blursize / 2;
    int channel=blockIdx.z; //obecny kanał
    extern __shared__ unsigned char shared_memory[];

    for(int i=threadIdx.y*blockDim.x+threadIdx.x;i< (blockDim.x + 2*radius)*(blockDim.y + 2*radius);i+=blockDim.x*blockDim.y)
    {
        int shared_x = i % (blockDim.x + 2*radius );
        int shared_y = i / (blockDim.x + 2*radius );
        int img_x = blockIdx.x * blockDim.x + shared_x - radius;
        int img_y = blockIdx.y * blockDim.y + shared_y - radius;

        // Handle border conditions
        img_x = min(max(img_x, 0), width - 1);
        img_y = min(max(img_y, 0), height - 1);

        shared_memory[i] = src[(img_y * width + img_x)* channels + channel];

    };
    __syncthreads();

    if(col<width && row<height)
    {
        double sum = 0.0;
        double weightSum = 0.0;
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                int shared_x = threadIdx.x + kx + radius;
                int shared_y = threadIdx.y + ky + radius;
                double weight = exp(-(kx * kx) / (2 * sigmaX * sigmaX) - (ky * ky) / (2 * sigmaY * sigmaY));
                sum += shared_memory[shared_y * (blockDim.x + blursize -1) + shared_x] * weight;
                weightSum += weight;
            }
        }
        dst[(row * width + col)*channels +channel] = static_cast<unsigned char>(sum / weightSum);
    }
}
torch::Tensor GaussianBlurCUDA(
    torch::Tensor img, 
    int blurSize, 
    double sigmaX, 
    double sigmaY)
{
    nvtxRangePushA("GaussianBlurCUDA - Start");
    TORCH_CHECK(img.is_cuda(), "Input image must be a CUDA tensor");
    TORCH_CHECK(img.dtype() == torch::kByte, "Input image must be of type Byte (unsigned char)");
    TORCH_CHECK(img.dim() == 3, "Input image must be a 3D tensor ( H, W, C)");

    int channels = img.size(2);
    int height = img.size(0);
    int width = img.size(1);
    int vector =channels*height*width;
    nvtxRangePushA("GaussianBlurCUDA - Memory Allocation");
    auto in_tensor = torch::empty({height, width, channels}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kByte));
    auto out_tensor = torch::empty_like(in_tensor);

    unsigned char* d_input = img.data_ptr<unsigned char>();
    unsigned char* d_in = in_tensor.data_ptr<unsigned char>();
    unsigned char* d_output = out_tensor.data_ptr<unsigned char>();

    cudaMemcpyAsync(
        d_in,
        d_input, 
        vector * sizeof(unsigned char), 
        cudaMemcpyDeviceToDevice, 
        at::cuda::getCurrentCUDAStream());

    nvtxRangePop(); // GaussianBlurCUDA - Memory Allocation
    nvtxRangePushA("GaussianBlurCUDA - Setup");
    
    dim3 blockSize=getOptimalBlockDim(width, height);
    dim3 gridSize((width + blockSize.x - 1)/blockSize.x, (height + blockSize.y -1)/blockSize.y, channels );

    using DeviceUchartPointer = thrust::device_ptr<unsigned char>;
    DeviceUchartPointer thrust_in_pointer = thrust::device_pointer_cast(d_in);
    DeviceUchartPointer thrust_out_pointer = thrust::device_pointer_cast(d_output);
    
    size_t sharedMemSize = (blockSize.x + blurSize ) * (blockSize.y + blurSize) * sizeof(unsigned char);
    nvtxRangePop(); // GaussianBlurCUDA - Setup
    nvtxRangePushA("GaussianBlurCUDA - Kernel Launch");
    GaussianBlurKernel<<<gridSize, blockSize, sharedMemSize, at::cuda::getCurrentCUDAStream()>>>(
        d_in,
        d_output,
        width,
        height,
        channels,
        blurSize,
        sigmaX,
        sigmaY);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    nvtxRangePop(); // GaussianBlurCUDA - Kernel Launch
    nvtxRangePop(); // GaussianBlurCUDA - Start
    return out_tensor;
}