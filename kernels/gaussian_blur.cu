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
namespace {
    int cachedKx = -1;
    int cachedKy = -1;
    float cachedSigmaX = -1.0f;
    float cachedSigmaY = -1.0f;
}

void computeGaussianCached(int kx, int ky, double sigmaX, double sigmaY)
{
    if (kx == cachedKx && ky == cachedKy &&
        fabs(sigmaX - cachedSigmaX) < 1e-5 &&
        fabs(sigmaY - cachedSigmaY) < 1e-5) {
        return;
    }

    cachedKx = kx;
    cachedKy = ky;
    cachedSigmaX = sigmaX;
    cachedSigmaY = sigmaY;

    int radiusX = kx / 2;
    int radiusY = ky / 2;
    int kernelSize = kx * ky;

    thrust::device_vector<float> d_kernel(kernelSize);
    auto begin = thrust::counting_iterator<int>(0);

    thrust::transform(
        thrust::device,
        begin, begin + kernelSize,
        d_kernel.begin(),
        [=] __device__ (int idx) {
            int x = idx % kx - radiusX;
            int y = idx / kx - radiusY;
            float gx = expf(-(x * x) / (2.0f * sigmaX * sigmaX));
            float gy = expf(-(y * y) / (2.0f * sigmaY * sigmaY));
            return gx * gy;
        }
    );

    float sum = thrust::reduce(
        thrust::device,
        d_kernel.begin(),
        d_kernel.end()
    );

    thrust::transform(
        thrust::device,
        d_kernel.begin(), d_kernel.end(),
        d_kernel.begin(),
        [=] __device__ (float v) { return v / sum; }
    );

    cudaMemcpyToSymbol(
        constGaussianKernel,
        thrust::raw_pointer_cast(d_kernel.data()),
        kernelSize * sizeof(float)
    );
}

__global__ void GaussianBlurKernel(
    unsigned char* src,
    unsigned char* dst,
    int width, int height,
    int channels,
    int kx, int ky)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z;

    int radiusX = kx / 2;
    int radiusY = ky / 2;

    extern __shared__ unsigned char shared_memory[];

    int sharedWidth  = blockDim.x + 2 * radiusX;
    int sharedHeight = blockDim.y + 2 * radiusY;

    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i < sharedWidth * sharedHeight;
         i += blockDim.x * blockDim.y)
    {
        int sx = i % sharedWidth;
        int sy = i / sharedWidth;

        int img_x = blockIdx.x * blockDim.x + sx - radiusX;
        int img_y = blockIdx.y * blockDim.y + sy - radiusY;

        img_x = min(max(img_x, 0), width  - 1);
        img_y = min(max(img_y, 0), height - 1);

        shared_memory[i] =
            src[(img_y * width + img_x) * channels + channel];
    }

    __syncthreads();

    if (col < width && row < height)
    {
        float sum = 0.0f;

        for (int y = -radiusY; y <= radiusY; ++y)
        {
            for (int x = -radiusX; x <= radiusX; ++x)
            {
                int sx = threadIdx.x + x + radiusX;
                int sy = threadIdx.y + y + radiusY;

                int k = (y + radiusY) * kx + (x + radiusX);

                float pixel = static_cast<float>(
                    shared_memory[sy * sharedWidth + sx]
                );

                sum += pixel * constGaussianKernel[k];
            }
        }

        dst[(row * width + col) * channels + channel] =
            static_cast<unsigned char>(sum);
    }
}

torch::Tensor GaussianBlurCUDA(
    torch::Tensor img,
    at::IntArrayRef ksize,
    double sigmaX,
    double sigmaY)
{
    int kx = ksize[0];
    int ky = ksize[1];
    nvtxRangePushA("GaussianBlurCUDA - Start");

    TORCH_CHECK(kx % 2 == 1 && ky % 2 == 1, "Kernel sizes must be odd");
    TORCH_CHECK(kx > 0 && ky > 0);
    TORCH_CHECK(kx <= maxblursize && ky <= maxblursize);

    int channels = img.size(2);
    int height   = img.size(0);
    int width    = img.size(1);

    nvtxRangePushA("GaussianBlurCUDA - Kernel Preparation");
    computeGaussianCached(kx, ky, sigmaX, sigmaY);
    nvtxRangePop();

    auto out_tensor = torch::empty_like(img);

    unsigned char* d_in  = img.data_ptr<unsigned char>();
    unsigned char* d_out = out_tensor.data_ptr<unsigned char>();

    nvtxRangePushA("GaussianBlurCUDA - Setup");

    dim3 blockSize = getOptimalBlockDim(width, height);
    dim3 gridSize(
        cdiv(width,  blockSize.x),
        cdiv(height, blockSize.y),
        channels
    );

    int radiusX = kx / 2;
    int radiusY = ky / 2;

    size_t sharedMemSize =
        (blockSize.x + 2 * radiusX) *
        (blockSize.y + 2 * radiusY) *
        sizeof(unsigned char);

    nvtxRangePop();

    nvtxRangePushA("GaussianBlurCUDA - Kernel Launch");

    GaussianBlurKernel<<<
        gridSize,
        blockSize,
        sharedMemSize,
        at::cuda::getCurrentCUDAStream()
    >>>(
        d_in,
        d_out,
        width,
        height,
        channels,
        kx,
        ky
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    nvtxRangePop();
    nvtxRangePop();

    return out_tensor;
}
