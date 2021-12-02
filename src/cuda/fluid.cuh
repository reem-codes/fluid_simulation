//
// Created by Reem on 29/11/2021.
//
#pragma once
// cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include <surface_functions.h>
#include "helper_cuda.h"

#define TILE_SIZE 16

// arrays for fluid simulation
__device__ float *velocity_x, *velocity_y, *pressure,  *divergence, *density, *red, *green, *blue, *temp;
__device__ curandState_t* states;
__device__ unsigned int NX, NY;

// enum types for easier access
// idea from here: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
enum ARRAY_TYPE {
    VELOCITY_X, VELOCITY_Y, PRESSURE, DENSITY, DIVERGENCE, TEMP, RED, GREEN, BLUE
};

// define functions used in both device and host
template <typename T>
__device__ __host__ T clamp(T x, T min_val, T max_val) {
    return  x < min_val ? min_val : x > max_val ? max_val : x;
}
__device__ __host__ float mix(float x, float y, float a);
__device__ __host__ float linear(float y0, float y1, float x0, float x1, float x);
__device__ __host__ float bilinear(float v11, float v21, float v12, float v22, float x, float x1, float x2, float y, float y1, float y2);


// functions used in host
void fluidSimulationInitialize(unsigned int sizex, unsigned int sizey);
void fluidSimulationReset();
void fluidSimulationDelete();
void seed(int i, int j, int radius, float power, float r, float g, float b);
void fluidSimulationStep(float timestep, unsigned int iter,  int integration, int solver,
                         float jitter, float visc, float speedX, float speedY);

void makeText(cudaSurfaceObject_t image);