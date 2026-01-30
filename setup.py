from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fastcv",
    ext_modules=[
        CUDAExtension(
            name="fastcv",
            sources=[
                "kernels/grayscale.cu",
                "kernels/box_blur.cu",
                "kernels/sobel.cu",
                "kernels/dilation.cu",
                "kernels/erosion.cu",
                "kernels/module.cpp",
                "kernels/gaussian_blur.cu",
            ],
            extra_compile_args={"cxx": ["-O2"], "nvcc" : ["--extended-lambda"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
