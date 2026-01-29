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

#define maxblursize 15
#define maxKernelSize ((maxblursize)*(maxblursize))

__constant__ float constGaussianKernel[maxKernelSize];

struct GaussianKernelFunctor {
    int radius;
    double sigmaX;
    double sigmaY;

    __device__
    float operator()(int idx) const {
        int x = idx % (2 * radius + 1) - radius;
        int y = idx / (2 * radius + 1) - radius;
        float value_x = exp(-(x * x) / (2 * sigmaX * sigmaX));
        float value_y = exp(-(y * y) / (2 * sigmaY * sigmaY));
        return value_x * value_y;
    }
};
struct NormalizeFunctor {
    float sum;

    __device__
    float operator()(float value) const {
        return value / sum;
    }
};

void computeGaussianKernel(int blurSize, double sigmaX, double sigmaY){
        int radius = blurSize / 2;
        int kernelSize = blurSize * blurSize;

        thrust::device_vector<float> d_kernel(kernelSize);
        auto begin = thrust::counting_iterator<int>(0);
        thrust::transform(
            thrust::device,
            begin, begin + kernelSize,
            d_kernel.begin(),
            GaussianKernelFunctor{radius, sigmaX, sigmaY}
        );
        // Normalize the kernel 
        float sum = thrust::reduce(
            thrust::device,
            d_kernel.begin(), 
            d_kernel.end());
        thrust::transform(
            thrust::device,
            d_kernel.begin(), d_kernel.end(),
            d_kernel.begin(),
            NormalizeFunctor{sum}
        );
        cudaMemcpyToSymbol(
            constGaussianKernel, 
            thrust::raw_pointer_cast(d_kernel.data()), 
            kernelSize * sizeof(float));
    };

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
    int channel=blockIdx.z; //current channel
    extern __shared__ unsigned char shared_memory[];

    for(int i=threadIdx.y*blockDim.x+threadIdx.x;i< (blockDim.x + blursize)*(blockDim.y + blursize);i+=blockDim.x*blockDim.y)
    {
        int shared_x = i % (blockDim.x + blursize );
        int shared_y = i / (blockDim.x + blursize );
        int img_x = blockIdx.x * blockDim.x + shared_x - radius;
        int img_y = blockIdx.y * blockDim.y + shared_y - radius;

        // Handle border conditions
        img_x = min(max(img_x, 0), width );
        img_y = min(max(img_y, 0), height);

        shared_memory[i] = src[(img_y * width + img_x)* channels + channel];

    };
    __syncthreads();

    if(col<width && row<height)
    {
        double sum = 0.0;

        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {

                int shared_x = threadIdx.x + kx + radius;
                int shared_y = threadIdx.y + ky + radius;
                int k = (ky+radius)*blursize + (kx+radius);
                float pixel= static_cast<float>(shared_memory[shared_y * (blockDim.x + blursize) + shared_x]);
                sum+=pixel * constGaussianKernel[k];
            }
        }
        
        dst[(row * width + col)*channels +channel] = static_cast<unsigned char>(sum);
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
    int radius = blurSize / 2;
    int channels = img.size(2);
    int height = img.size(0);
    int width = img.size(1);
    nvtxRangePushA("GaussianBlurCUDA - Kernel Preparation");
    computeGaussianKernel(
        blurSize, 
        static_cast<float>(sigmaX), 
        static_cast<float>(sigmaY));
    auto out_tensor = torch::empty_like(img);

    
    unsigned char* d_in = img.data_ptr<unsigned char>();
    unsigned char* d_output = out_tensor.data_ptr<unsigned char>();

    

    nvtxRangePop(); // GaussianBlurCUDA - Memory Allocation
    nvtxRangePushA("GaussianBlurCUDA - Setup");
    
    dim3 blockSize=getOptimalBlockDim(width, height);
    dim3 gridSize(
        cdiv(width, blockSize.x),
        cdiv(height, blockSize.y), 
        channels 
        );

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
    cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) {
            printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
}
    cudaDeviceSynchronize();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    nvtxRangePop(); // GaussianBlurCUDA - Kernel Launch
    nvtxRangePop(); // GaussianBlurCUDA - Start
    return out_tensor;
}