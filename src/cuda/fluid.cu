//
// Created by Reem on 29/11/2021.
//
#include "fluid.cuh"
#include <cstdio>


unsigned int nx, ny, size, byteSize, byteSizeP;
dim3 dimGrid, dimBlock;
int gx, gy;
float* data;

template<ARRAY_TYPE type>
void init() {
    /** initialize the global device arrays
     * can't use cudamalloc() directly because it's a device address,
     * so use cudaMemcpyToSymbol instead
     */
    float* d_data;
    checkCudaErrors(cudaMalloc(&d_data, byteSize));

    checkCudaErrors(cudaMemcpy(d_data, data, byteSize, cudaMemcpyHostToDevice));

    // allocate memory in GPU
    if (type == VELOCITY_X) {
        checkCudaErrors(cudaMemcpyToSymbol(velocity_x, &d_data, byteSizeP));
    }
    else if(type == VELOCITY_Y) {
        checkCudaErrors(cudaMemcpyToSymbol(velocity_y, &d_data, byteSizeP));
    }
    else if(type == PRESSURE) {
        checkCudaErrors(cudaMemcpyToSymbol(pressure, &d_data, byteSizeP));
    }
    else if(type == DENSITY) {
        checkCudaErrors(cudaMemcpyToSymbol(density, &d_data, byteSizeP));
    }
    else if(type == DIVERGENCE) {
        checkCudaErrors(cudaMemcpyToSymbol(divergence, &d_data, byteSizeP));
    }
    else if(type == TEMP) {
        checkCudaErrors(cudaMemcpyToSymbol(temp, &d_data, byteSizeP));
    }
    else if(type == RED) {
        checkCudaErrors(cudaMemcpyToSymbol(red, &d_data, byteSizeP));
    }
    else if(type == GREEN) {
        checkCudaErrors(cudaMemcpyToSymbol(green, &d_data, byteSizeP));
    }
    else if(type == BLUE) {
        checkCudaErrors(cudaMemcpyToSymbol(blue, &d_data, byteSizeP));
    }
}
__global__ void initRand() {
    /**
     * initialize cuRand, a library for random number generator in CUDA
     * for more details:
     * https://docs.nvidia.com/cuda/curand/device-api-overview.html
     * http://ianfinlayson.net/class/cpsc425/notes/cuda-random
     */
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < NX && j < NY) {
        unsigned int index = i + j * NX;
        curand_init(32, index, 0, &states[index]);
    }
}
void fluidSimulationInitialize(unsigned int sizex, unsigned int sizey) {
    // initialize variables in host
    nx = sizex;
    ny = sizey;
    size = nx * ny;
    byteSize = size * sizeof (float);
    byteSizeP = sizeof (float*);

    // used to set global device arrays to zero
    data = new float[size];
    for (int i = 0; i < size; i++)
        data[i] = 0;

    printf("Grid size: %d times %d = %d, %d bytes\n", nx, ny, size, byteSize);

    // calculate thread sizes
    dimGrid = dim3(ceil((float)nx/TILE_SIZE), ceil((float)ny/TILE_SIZE), 1);
    dimBlock = dim3(TILE_SIZE, TILE_SIZE, 1);
    printf("(%d, %d) blocks, (%d, %d) threads each\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
    // gridsize for boundary x and y
    gx = ceil(((float)(nx-2)/TILE_SIZE));
    gy = ceil(((float)(ny-2)/TILE_SIZE));

    // initialize GPU global device arrays
    init<VELOCITY_X>();
    init<VELOCITY_Y>();
    init<PRESSURE>();
    init<DENSITY>();
    init<DIVERGENCE>();
    init<TEMP>();
    init<RED>();
    init<GREEN>();
    init<BLUE>();

    // initialize NX and NY in the GPU
    unsigned int *temp_nx, *temp_ny;
    checkCudaErrors(cudaMalloc(&temp_nx, sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(temp_nx, &nx, sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(NX, temp_nx, sizeof(unsigned int)));

    checkCudaErrors(cudaMalloc(&temp_ny, sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(temp_ny, &ny, sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(NY, temp_ny, sizeof(unsigned int)));

    // initialize curand random states
    curandState_t* d_data;
    checkCudaErrors(cudaMalloc(&d_data, size * sizeof(curandState_t)));
    checkCudaErrors(cudaMemcpyToSymbol(states, &d_data, sizeof(curandState_t*)));
    initRand<<<dimGrid, dimBlock>>>();
}
template<ARRAY_TYPE type>
__global__ void reset() {
    /**
     * reset GPU global device arrays to zero
     */
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < NX && j < NY) {
        unsigned int index = i + j * NX;
        if (type == VELOCITY_X)
            velocity_x[index] = 0;
        else if (type == VELOCITY_Y)
            velocity_y[index] = 0;
        else if (type == PRESSURE)
            pressure[index] = 0;
        else if (type == DENSITY)
            density[index] = 0;
        else if (type == DIVERGENCE)
            divergence[index] = 0;
        else if (type == TEMP)
            temp[index] = 0;
        else if (type == RED)
            red[index] = 0;
        else if (type == GREEN)
            green[index] = 0;
        else if (type == BLUE)
            blue[index] = 0;
    }
}
void fluidSimulationReset() {
    /**
     * reset GPU global device arrays to zero
     */
    reset<VELOCITY_X><<<dimGrid, dimBlock>>>();
    reset<VELOCITY_Y><<<dimGrid, dimBlock>>>();
    reset<PRESSURE><<<dimGrid, dimBlock>>>();
    reset<DENSITY><<<dimGrid, dimBlock>>>();
    reset<DIVERGENCE><<<dimGrid, dimBlock>>>();
    reset<TEMP><<<dimGrid, dimBlock>>>();
    reset<RED><<<dimGrid, dimBlock>>>();
    reset<GREEN><<<dimGrid, dimBlock>>>();
    reset<BLUE><<<dimGrid, dimBlock>>>();
    checkCudaErrors(cudaDeviceSynchronize());
}
void fluidSimulationDelete() {
    /**
     * Delete resources
     */
    checkCudaErrors(cudaFree(velocity_x));
    checkCudaErrors(cudaFree(velocity_y));
    checkCudaErrors(cudaFree(pressure));
    checkCudaErrors(cudaFree(density));
    checkCudaErrors(cudaFree(divergence));
    checkCudaErrors(cudaFree(temp));
    checkCudaErrors(cudaFree(red));
    checkCudaErrors(cudaFree(green));
    checkCudaErrors(cudaFree(blue));
    delete[] data;
}
template<ARRAY_TYPE type>
__global__ void swap() {
    /** swap the specified array and TEMP
     * used because solving fluid equations often needs double buffering
     */
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < NX && j < NY) {

        unsigned int index = i + j * NX;
        if (type == VELOCITY_X)
            velocity_x[index] = temp[index];
        else if (type == VELOCITY_Y)
            velocity_y[index] = temp[index];
        else if (type == PRESSURE)
            pressure[index] = temp[index];
        else if (type == DENSITY)
            density[index] = temp[index];
        else if (type == DIVERGENCE)
            divergence[index] = temp[index];
        else if (type == TEMP)
            temp[index] = temp[index];
        else if (type == RED)
            red[index] = temp[index];
        else if (type == GREEN)
            green[index] = temp[index];
        else if (type == BLUE)
            blue[index] = temp[index];
    }
}
void fluidSimulationSwap(ARRAY_TYPE type){
    /** swap the specified array and TEMP
     * used because solving fluid equations often needs double buffering
     */
    if(type == VELOCITY_X)
        swap<VELOCITY_X><<<dimGrid, dimBlock>>>();
    else if(type == VELOCITY_Y)
        swap<VELOCITY_Y><<<dimGrid, dimBlock>>>();
    else if(type == PRESSURE)
        swap<PRESSURE><<<dimGrid, dimBlock>>>();
    else if(type == DENSITY)
        swap<DENSITY><<<dimGrid, dimBlock>>>();
    else if(type == DIVERGENCE)
        swap<DIVERGENCE><<<dimGrid, dimBlock>>>();
    else if(type == TEMP)
        swap<TEMP><<<dimGrid, dimBlock>>>();
    else if(type == RED)
        swap<RED><<<dimGrid, dimBlock>>>();
    else if(type == GREEN)
        swap<GREEN><<<dimGrid, dimBlock>>>();
    else if(type == BLUE)
        swap<BLUE><<<dimGrid, dimBlock>>>();
    checkCudaErrors(cudaDeviceSynchronize());
}

__device__ void colorUpdate(float* color, int index, float c, float d) {
    /** update a color given parameters **/
    color[index] += clamp(density[index], 0.f, 100.f) * c;
}
__device__ void halfCircle(int index, float distance, float power, float r, float g, float b) {
    /** update colors and density variables,
     * made into a function because I have two halves of a circle to update **/
    density[index] += power / (distance + 1);
    colorUpdate(red, index, r, distance);
    colorUpdate(green, index, g, distance);
    colorUpdate(blue, index, b, distance);
}
__global__ void seedColor(int x, int y, int radius, float power, float r, float g, float b) {
    /** given x, y position,
     * add density in the shape of the circle
     * and proportional to the distance from the radius and the given power
     */
    float distance;
    // from 0-2r, we want y-r to y+r,
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x + y - radius;
    if(j > 0 && j < NY - 2) {
        for (int i = x, k=x+1; (i-x)*(i-x)+(j-y)*(j-y) <= radius * radius; i--, k++) {
            distance = sqrtf((i-x)*(i-x) + (j-y)*(j-y));
            if(i > 0 && i < NX - 2) { // left half of circle
                halfCircle(i + j * NX, distance, power, r, g, b);
            }
            if(k > 0 && k < NX - 2) { // right half of circle
                halfCircle(k + j * NX, distance, power, r, g, b);
            }
        }
    }
}

void seed(int i, int j, int radius, float power, float r, float g, float b) {
    /** given x, y position,
     * add density in the shape of the circle
     * and proportional to the distance from the radius and the given power
     */
    seedColor<<<1, 2 * radius>>>(i, j, radius,  power, r, g, b);
    checkCudaErrors(cudaDeviceSynchronize());
}

__device__ __host__ float mix(float x, float y, float a) {
    /** opeGL mix function */
    return (1 - a) * x + a * y;
}
__device__ __host__ float linear(float y0, float y1, float x0, float x1, float x) {
    /** linear interpolation */
    // y = (y0 (x1 - x) + y1 (x - x0)) / (x1 - x0)
    // https://en.wikipedia.org/wiki/Linear_interpolation
    return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0);
}
__device__ __host__ float bilinear(float v11, float v21, float v12, float v22, float x, float x1, float x2, float y, float y1, float y2) {
    /** bilinear interpolation */
    // (Q11(x2-x)(y2-y)+Q21(x-x1)(y2-y)+Q12(x2-x)(y-y1)+Q22(x-x1)(y-y1)) / (x2-x1)(y2-y1)
    // https://en.wikipedia.org/wiki/Bilinear_interpolation
    return (v11 * (x2 - x ) * (y2 - y ) +
            v21 * (x  - x1) * (y2 - y ) +
            v12 * (x2 - x ) * (y  - y1) +
            v22 * (x  - x1) * (y  - y1))/ (x2 - x1) * (y2 - y1);
}
__device__ float interpolate(float* data, float x, float y) {
    /** Interpolation is done using:
     * bilinear interpolation if inside grid
     * linear interpolation if in the sides
     * nearest neighbor if outside the grid */

    int x1 = clamp((int)floorf(x), 0, (int)NX-1);
    int x2 = clamp((int)ceilf(x), 0, (int)NX-1);
    int y1 = clamp((int)floorf(y), 0, (int)NX-1);
    int y2 = clamp((int)ceilf(y), 0, (int)NX-1);

    // case 1: nearest neighbor
    if (x1 == x2 && y1 == y2)
        return data[x1 + y1 * NX];

    // case 2: linear interpolation
    if (x1 == x2 && y1 != y2)
        return linear(data[x1 + y1 * NX], data[x1 + y2 * NX], y1, y2, y);
    if (y1 == y2 && x1 != x2)
        return linear(data[x1 + y1 * NX], data[x2 + y1 * NX], x1, x2, x);

    // case 3: bilinear interpolation
    return bilinear(data[x1 + y1 * NX], data[x2 + y1 * NX], data[x1 + y2 * NX], data[x2 + y2 * NX], x, x1, x2, y, y1, y2);
}

__device__ void euler(float timestep, int index, int i, int j, float& x, float& y) {
    /** Euler forward integration to calculate the new position
     * x_{i+1} = x_{i} + v(x_{i}) * dt
     */
    x = (float)i - timestep * velocity_x[index];
    y = (float)j - timestep * velocity_y[index];
}
__device__ void rk2(float timestep, int index, int i, int j, float& x, float& y) {
    /** Runge-Kutta-2 integration to calculate the new position
     * x_{i+1} = x_{i} + v(x_{i} + v(x_{i}) * dt / 2) * dt
     */
    float u, v;
    x = (float)i - velocity_x[index] * timestep / 2;
    y = (float)j - velocity_y[index] * timestep / 2;
    u = interpolate(velocity_x, x, y);
    v = interpolate(velocity_y, x, y);
    x = (float)i - u * timestep;
    y = (float)j - v * timestep;
}
__device__ void rk4(float timestep, int i, int j, float& x, float& y) {
    /** Runge-Kutta-4 integration to calculate the new position
     * x_{i+1} = x_{i} + (a + 2 b + 2 c + d) / 6
     * a = v(x_{i}) * dt
     * b = v(x_{i} + a/2) * dt
     * c = v(x_{i} + b/2) * dt
     * d = v(x_{i} + c) * dt
     */
    float a_x, a_y, b_x, b_y, c_x, c_y;
    a_x = interpolate(velocity_x, (float)i, (float)j) * timestep;
    a_y = interpolate(velocity_y, (float)i, (float)j) * timestep;

    b_x = interpolate(velocity_x, (float)i-a_x/2, (float)j-a_y/2) * timestep;
    b_y = interpolate(velocity_y, (float)i-a_x/2, (float)j-a_y/2) * timestep;

    c_x = interpolate(velocity_x, (float)i-b_x/2, (float)j-b_y/2) * timestep;
    c_y = interpolate(velocity_y, (float)i-b_x/2, (float)j-b_y/2) * timestep;

    x = interpolate(velocity_x, (float)i-c_x, (float)j-c_y) * timestep;
    y = interpolate(velocity_y, (float)i-c_x, (float)j-c_y) * timestep;

    x = (float)i - (a_x + 2 * b_x + 2 * c_x + x) / 6;
    y = (float)j - (a_y + 2 * b_y + 2 * c_y + y) / 6;
}

template<ARRAY_TYPE type, int integration>
__global__ void advect_kernel(float timestep) {
    /**
     * Given some quantity Q on our simulation grid, how will Q change ∆t later?
     * q(x, t+dt) = q(integrate(x, velocity(x, t), dt) t)
       for each grid cell i, j, k
           calculate -gradient Q
           calculate spatial Q i j k to store in X
           calculate X prev = X - gradient * timestep
           set gridpoint for Q n + 1 nearest to X prev = Q i j k
       set Q = Q n + 1
    */
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < NX && j < NY) {
        unsigned int index = i + j * NX;
        float x, y;
        if (integration == 1)
            rk2(timestep, index, i, j, x, y);
        if (integration == 2)
            rk4(timestep, i, j, x, y);
        else
            euler(timestep, index, i, j, x, y);

        if (type == VELOCITY_X)
            temp[index] = interpolate(velocity_x, x, y);
        else if (type == VELOCITY_Y)
            temp[index] = interpolate(velocity_y, x, y);
        else if (type == DENSITY)
            temp[index] = interpolate(density, x, y);
        else if (type == RED)
            temp[index] = interpolate(red, x, y);
        else if (type == GREEN)
            temp[index] = interpolate(green, x, y);
        else if (type == BLUE)
            temp[index] = interpolate(blue, x, y);
    }
}

void advect(ARRAY_TYPE type, int integration, float timestep) {
    /**
     * Given some quantity Q on our simulation grid, how will Q change ∆t later?
     */
    if (integration == 1) { // RK2
        if(type == VELOCITY_X)
            advect_kernel<VELOCITY_X, 1><<<dimGrid, dimBlock>>>(timestep);
        else if(type == VELOCITY_Y)
            advect_kernel<VELOCITY_Y, 1><<<dimGrid, dimBlock>>>(timestep);
        else if(type == DENSITY)
            advect_kernel<DENSITY, 1><<<dimGrid, dimBlock>>>(timestep);
        else if(type == RED)
            advect_kernel<RED, 1><<<dimGrid, dimBlock>>>(timestep);
        else if(type == GREEN)
            advect_kernel<GREEN, 1><<<dimGrid, dimBlock>>>(timestep);
        else if(type == BLUE)
            advect_kernel<BLUE, 1><<<dimGrid, dimBlock>>>(timestep);
    }
    else if (integration == 2) { // RK4
        if(type == VELOCITY_X)
            advect_kernel<VELOCITY_X, 2><<<dimGrid, dimBlock>>>(timestep);
        else if(type == VELOCITY_Y)
            advect_kernel<VELOCITY_Y, 2><<<dimGrid, dimBlock>>>(timestep);
        else if(type == DENSITY)
            advect_kernel<DENSITY, 2><<<dimGrid, dimBlock>>>(timestep);
        else if(type == RED)
            advect_kernel<RED, 2><<<dimGrid, dimBlock>>>(timestep);
        else if(type == GREEN)
            advect_kernel<GREEN, 2><<<dimGrid, dimBlock>>>(timestep);
        else if(type == BLUE)
            advect_kernel<BLUE, 2><<<dimGrid, dimBlock>>>(timestep);
    }
    else { // EULER
        if(type == VELOCITY_X)
            advect_kernel<VELOCITY_X, 0><<<dimGrid, dimBlock>>>(timestep);
        else if(type == VELOCITY_Y)
            advect_kernel<VELOCITY_Y, 0><<<dimGrid, dimBlock>>>(timestep);
        else if(type == DENSITY)
            advect_kernel<DENSITY, 0><<<dimGrid, dimBlock>>>(timestep);
        else if(type == RED)
            advect_kernel<RED, 0><<<dimGrid, dimBlock>>>(timestep);
        else if(type == GREEN)
            advect_kernel<GREEN, 0><<<dimGrid, dimBlock>>>(timestep);
        else if(type == BLUE)
            advect_kernel<BLUE, 0><<<dimGrid, dimBlock>>>(timestep);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    fluidSimulationSwap(type);
}

__global__ void force(float jitter, float speedX, float speedY, float timestep) {
    /** apply force to the field
     * ie change velocity
     */

    float r;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < NX && j < NY) {

        unsigned int index = i + j * NX;

        // random jittering around density
        r = curand_uniform(&states[index]);
        velocity_x[index] += mix(-jitter, jitter, r);
        r = curand_uniform(&states[index]);
        velocity_y[index] += mix(-jitter, jitter, r);

        // change speed
        velocity_x[index] += speedX * density[index] * 0.1f;
        velocity_y[index] += speedY * density[index] * 0.1f;

        // decay with time
//    velocity_x[index] *= 0.999f;
//    velocity_y[index] *= 0.999f;
//    density[index] *= 0.999f;
    }
}

__device__ float computeGradient(float* data, int i, int j, int axis) {
    /**
     * compute gradient. axis = 0 -> x, otherwise y
     * use central difference everywhere, except at borders. use forward/backward difference

     * Case 1: inside the grid - central difference
         g x = 0.5 * (f(x + 1, y) – f(x - 1, y))
         g y = 0.5 * (f(x, y + 1) – f(x, y - 1))
     * case 2: at left/bottom border - forward difference
         g x = f(x + 1, y) - f(x, y)
         g y = f(x, y + 1) - f(x, y)
     * case 3: at right/top border - backward difference
         g x = f(x, y) - f(x - 1, y)
         g y = f(x, y) - f(x, y - 1)
     */

    if(axis == 0){ // partial w.r.t. i
        if(i > 0 && i < (NX - 1)) // central difference
            return 0.5f * (data[(i + 1) + j * NX] - data[(i - 1) + j * NX]);
        if(i == 0) // forward difference
            return data[(i + 1) + j * NX] - data[i + j * NX];
        else  // backward difference
            return data[i + j * NX] - data[(i - 1) + j * NX];
    } else { // partial w.r.t. j
        if(j > 0 && j < (NY - 1))  // central difference
            return 0.5f * (data[i + (j + 1) * NX] - data[i + (j - 1) * NX]);
        if(j == 0) // forward difference
            return data[i + (j + 1) * NX] - data[i + j * NX];
        else // backward difference
            return data[i + j * NX] - data[i + (j - 1) * NX];
    }
}
__device__ float computeLaplacian(float* data, int i, int j) {
    /** laplacian is computed as
     * 0.25 * (x(i+1, j) + x(i-1, j) + x(i, j+1) + x(i, j-1))
     **/
    float sum = 0;
    if (j > 0) {
        sum += data[i + (j - 1) * NX];
    }
    if (i > 0) {
        sum += data[(i - 1) + j * NX];
    }
    if (j < NY - 2) {
        sum += data[i + (j + 1) * NX];
    }
    if (i < NX - 2) {
        sum += data[(i + 1) + j * NX];
    }
    return sum;
}

template<int solver, ARRAY_TYPE X>
__global__ void iteration(float alpha, float beta) {
    /** a single jacobi/gauss seidel iteration
     * to solve the poisson equations for pressure and viscosity solve
     */
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < NX && j < NY) {
        unsigned int index = i + j * NX;
        if (solver == 0) { // if jacobi
            if (X == PRESSURE)
                temp[index] = (alpha * computeLaplacian(pressure, i, j) + divergence[index]) * beta;
            else if (X == VELOCITY_X)
                temp[index] = (alpha * computeLaplacian(velocity_x, i, j) + velocity_x[index]) * beta;
            else if (X == VELOCITY_Y)
                temp[index] = (alpha * computeLaplacian(velocity_y, i, j) + velocity_y[index]) * beta;
            else if (X == DENSITY)
                temp[index] = (alpha * computeLaplacian(density, i, j) + density[index]) * beta;
            else if (X == RED) temp[index] = (alpha * computeLaplacian(red, i, j) + red[index]) * beta;
            else if (X == GREEN) temp[index] = (alpha * computeLaplacian(green, i, j) + green[index]) * beta;
            else if (X == BLUE) temp[index] = (alpha * computeLaplacian(blue, i, j) + blue[index]) * beta;
        } else { // gauss-seidel
            if (X == PRESSURE)
                pressure[index] = (alpha * computeLaplacian(pressure, i, j) + divergence[index]) * beta;
            else if (X == VELOCITY_X)
                velocity_x[index] = (alpha * computeLaplacian(velocity_x, i, j) + velocity_x[index]) * beta;
            else if (X == VELOCITY_Y)
                velocity_y[index] = (alpha * computeLaplacian(velocity_y, i, j) + velocity_y[index]) * beta;
            else if (X == DENSITY)
                density[index] = (alpha * computeLaplacian(density, i, j) + density[index]) * beta;
            else if (X == RED) red[index] = (alpha * computeLaplacian(red, i, j) + red[index]) * beta;
            else if (X == GREEN) green[index] = (alpha * computeLaplacian(green, i, j) + green[index]) * beta;
            else if (X == BLUE) blue[index] = (alpha * computeLaplacian(blue, i, j) + blue[index]) * beta;
        }
    }
}
void jacobi(ARRAY_TYPE x, float alpha, float beta, unsigned int iter) {
    /** Jacobi iteration for solving systems of linear equations
     * Ax = b, A=D+L+U
     * x(k+1) = D^-1 (b - (L+U)x(k))
     * x_i(k+1) = (b_i - sum(j=1, j!=i, n){a_ij x_j(k)}) / a_ii
     * https://en.wikipedia.org/wiki/Jacobi_method
     * */
    for(int k = 0; k < iter; k++) {
        if(x == PRESSURE) {
            iteration<0, PRESSURE><<<dimGrid, dimBlock>>>(alpha, beta);
            checkCudaErrors(cudaDeviceSynchronize());
            fluidSimulationSwap(PRESSURE);
        }
        else if(x == VELOCITY_X) {
            iteration<0, VELOCITY_X><<<dimGrid, dimBlock>>>(alpha, beta);
            checkCudaErrors(cudaDeviceSynchronize());
            fluidSimulationSwap(VELOCITY_X);
        }
        else if(x == VELOCITY_Y) {
            iteration<0, VELOCITY_Y><<<dimGrid, dimBlock>>>(alpha, beta);
            checkCudaErrors(cudaDeviceSynchronize());
            fluidSimulationSwap(VELOCITY_Y);
        }
        else if(x == DENSITY) {
            iteration<0, DENSITY><<<dimGrid, dimBlock>>>(alpha, beta);
            checkCudaErrors(cudaDeviceSynchronize());
            fluidSimulationSwap(DENSITY);
        }
        else if(x == RED) {
            iteration<0, RED><<<dimGrid, dimBlock>>>(alpha, beta);
            checkCudaErrors(cudaDeviceSynchronize());
            fluidSimulationSwap(RED);
        }
        else if(x == GREEN) {
            iteration<0, GREEN><<<dimGrid, dimBlock>>>(alpha, beta);
            checkCudaErrors(cudaDeviceSynchronize());
            fluidSimulationSwap(GREEN);
        }
        else if(x == BLUE) {
            iteration<0, BLUE><<<dimGrid, dimBlock>>>(alpha, beta);
            checkCudaErrors(cudaDeviceSynchronize());
            fluidSimulationSwap(BLUE);
        }
    }
}
void gaussSeidel(ARRAY_TYPE x, float alpha, float beta, unsigned int iter) {
    /** gauss-seidel iteration for solving systems of linear equations
     * Ax = b, A=LU
     * x(k+1) = L^-1 (b - Ux(k))
     * x_i(k+1) = (b_i - sum(j=1, i-1){a_ij x_j(k+1)} - sum(j=i+1, n){a_ij x_j(k)}) / a_ii
     * https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
     * */
    for(int k = 0; k < iter; k++) {
        if(x == PRESSURE) {
            iteration<1, PRESSURE><<<dimGrid, dimBlock>>>(alpha, beta);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        else if(x == VELOCITY_X) {
            iteration<1, VELOCITY_X><<<dimGrid, dimBlock>>>(alpha, beta);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        else if(x == VELOCITY_Y) {
            iteration<1, VELOCITY_Y><<<dimGrid, dimBlock>>>(alpha, beta);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        else if(x == DENSITY) {
            iteration<1, DENSITY><<<dimGrid, dimBlock>>>(alpha, beta);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        else if(x == RED) {
            iteration<1, RED><<<dimGrid, dimBlock>>>(alpha, beta);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        else if(x == GREEN) {
            iteration<1, GREEN><<<dimGrid, dimBlock>>>(alpha, beta);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        else if(x == BLUE) {
            iteration<1, BLUE><<<dimGrid, dimBlock>>>(alpha, beta);
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }
}
void diffuse(ARRAY_TYPE type, float visc, unsigned int iter, int solver, float timestep) {
    /**
     * for solving viscosity term.
     * jacobi or gauss-seidel are used to solve the Poisson equation of viscosity
     */

    float a = visc * timestep;
    if (solver == 0) {
        if(type == VELOCITY_X) jacobi(VELOCITY_X, a, 1.f / (1.f + 4.f * a), iter);
        else if(type == VELOCITY_Y) jacobi(VELOCITY_Y, a, 1.f / (1.f + 4.f * a), iter);
        else if(type == DENSITY) jacobi(DENSITY, a, 1.f / (1.f + 4.f * a), iter);
        else if(type == RED) jacobi(RED, a, 1.f / (1.f + 4.f * a), iter);
        else if(type == GREEN) jacobi(GREEN, a, 1.f / (1.f + 4.f * a), iter);
        else if(type == BLUE) jacobi(BLUE, a, 1.f / (1.f + 4.f * a), iter);
    }
    else {
        if(type == VELOCITY_X) gaussSeidel(VELOCITY_X, a, 1.f / (1.f + 4.f * a), iter);
        else if(type == VELOCITY_Y) gaussSeidel(VELOCITY_Y, a, 1.f / (1.f + 4.f * a), iter);
        else if(type == DENSITY) gaussSeidel(DENSITY, a, 1.f / (1.f + 4.f * a), iter);
        else if(type == RED) gaussSeidel(RED, a, 1.f / (1.f + 4.f * a), iter);
        else if(type == GREEN) gaussSeidel(GREEN, a, 1.f / (1.f + 4.f * a), iter);
        else if(type == BLUE) gaussSeidel(BLUE, a, 1.f / (1.f + 4.f * a), iter);
    }
}

__global__ void computeDivergence() {
    /**
     * divergence is  du/dx + dv/dy
     * negative divergence for pressure solve
     */
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NX && j < NY) {
        unsigned int index = i + j * NX;
        divergence[index] = -(computeGradient(velocity_x, i, j, 0) +
                              computeGradient(velocity_y, i, j, 1));
    }
}

__global__ void subtractPressureGradient() {
    /** update velocity by subtracting grad pressure
     * based on Helmholtz-Hodge Decomposition Theorem
     */
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < NX && j < NY) {
        unsigned int index = i + j * NX;
        velocity_x[index] -= computeGradient(pressure, i, j, 0);
        velocity_y[index] -= computeGradient(pressure, i, j, 1);
    }
}
void computePressure(int solver, unsigned int iter) {
    /**
     * compute the divergence
     * jacobi/gauss-seidel to find grad pressure using pressure laplacian - divergence
     * update vector field by subtracting velocity from pressure
     */
    reset<PRESSURE><<<dimGrid, dimBlock>>>();

    computeDivergence<<<dimGrid, dimBlock>>>();
    checkCudaErrors(cudaDeviceSynchronize());

    if(solver == 0)
        jacobi(PRESSURE, 1, 0.25f, iter);
    else
        gaussSeidel(PRESSURE, 1, 0.25f, iter);

    subtractPressureGradient<<<dimGrid, dimBlock>>>();
    checkCudaErrors(cudaDeviceSynchronize());
}

template<ARRAY_TYPE type, int dir>
__global__ void boundary_kernel() {
    /**
     * to satisfy the conditions, the boundary values should be:
     * same value as the neighboring cell for pressure
     * the negative of the neighboring cell value for velocity
     */
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // [1, n-1) instead of [0, n)
    if(i < NX - 1) {
        if (dir == 1) { // horizontal
            if (type == PRESSURE) {
                pressure[i + 0 * NX] = pressure[i + 1 * NX];
                pressure[i + (NY - 1) * NX] = pressure[i + (NY - 2) * NX];
            } else if (type == VELOCITY_X) {
                velocity_x[i + 0 * NX] = -velocity_x[i + 1 * NX];
                velocity_x[i + (NY - 1) * NX] = -velocity_x[i + (NY - 2) * NX];
            } else if (type == VELOCITY_Y) {
                velocity_y[i + 0 * NX] = -velocity_y[i + 1 * NX];
                velocity_y[i + (NY - 1) * NX] = -velocity_y[i + (NY - 2) * NX];
            }
        } else if (dir == 2) { // vertical
            if (type == PRESSURE) {
                pressure[0 + i * NX] = pressure[1 + i * NX];
                pressure[(NX - 1) + i * NX] = pressure[(NX - 2) + i * NX];
            } else if (type == VELOCITY_X) {
                velocity_x[0 + i * NX] = -velocity_x[1 + i * NX];
                velocity_x[(NX - 1) + i * NX] = -velocity_x[(NX - 2) + i * NX];
            } else if (type == VELOCITY_Y) {
                velocity_y[0 + i * NX] = -velocity_y[1 + i * NX];
                velocity_y[(NX - 1) + i * NX] = -velocity_y[(NX - 2) + i * NX];
            }
        } else { // corners
            if (type == PRESSURE) {
                pressure[0 + 0 * NX] = 0.5f * (pressure[1 + 0 * NX] + pressure[0 + 1 * NX]);
                pressure[0 + (NY - 1) * NX] = 0.5f * (pressure[1 + (NY - 1) * NX] + pressure[0 + (NY - 2) * NX]);
                pressure[(NX - 1) + 0 * NX] = 0.5f * (pressure[(NX - 1) + 1 * NX] + pressure[(NX - 2) + 0 * NX]);
                pressure[(NX - 1) + (NY - 1) * NX] =
                        0.5f * (pressure[(NX - 2) + (NY - 1) * NX] + pressure[(NX - 1) + (NY - 2) * NX]);
            } else if (type == VELOCITY_X) {
                velocity_x[0 + 0 * NX] = 0.5f * (velocity_x[1 + 0 * NX] + velocity_x[0 + 1 * NX]);
                velocity_x[0 + (NY - 1) * NX] = 0.5f * (velocity_x[1 + (NY - 1) * NX] + velocity_x[0 + (NY - 2) * NX]);
                velocity_x[(NX - 1) + 0 * NX] = 0.5f * (velocity_x[(NX - 1) + 1 * NX] + velocity_x[(NX - 2) + 0 * NX]);
                velocity_x[(NX - 1) + (NY - 1) * NX] =
                        0.5f * (velocity_x[(NX - 2) + (NY - 1) * NX] + velocity_x[(NX - 1) + (NY - 2) * NX]);
            } else if (type == VELOCITY_Y) {
                velocity_y[0 + 0 * NX] = 0.5f * (velocity_y[1 + 0 * NX] + velocity_y[0 + 1 * NX]);
                velocity_y[0 + (NY - 1) * NX] = 0.5f * (velocity_y[1 + (NY - 1) * NX] + velocity_y[0 + (NY - 2) * NX]);
                velocity_y[(NX - 1) + 0 * NX] = 0.5f * (velocity_y[(NX - 1) + 1 * NX] + velocity_y[(NX - 2) + 0 * NX]);
                velocity_y[(NX - 1) + (NY - 1) * NX] =
                        0.5f * (velocity_y[(NX - 2) + (NY - 1) * NX] + velocity_y[(NX - 1) + (NY - 2) * NX]);
            }
        }
    }
}

void setBoundary(ARRAY_TYPE type) {
    /**
     * fluid can't escape the box
     * no-slip (zero) velocity boundary condition
     * pure Neumann pressure boundary condition
     */
    if(type == PRESSURE) {
        boundary_kernel<PRESSURE, 1><<<gx, TILE_SIZE>>>();
        boundary_kernel<PRESSURE, 2><<<gy, TILE_SIZE>>>();
        checkCudaErrors(cudaDeviceSynchronize());
        boundary_kernel<PRESSURE, 0><<<1, 1>>>();
    } else if(type == VELOCITY_X) {
        boundary_kernel<VELOCITY_X, 1><<<gx, TILE_SIZE>>>();
        boundary_kernel<VELOCITY_X, 2><<<gy, TILE_SIZE>>>();
        checkCudaErrors(cudaDeviceSynchronize());
        boundary_kernel<VELOCITY_X, 0><<<1, 1>>>();
    } else if(type == VELOCITY_Y) {
        boundary_kernel<VELOCITY_Y, 1><<<gx, TILE_SIZE>>>();
        boundary_kernel<VELOCITY_Y, 2><<<gy, TILE_SIZE>>>();
        checkCudaErrors(cudaDeviceSynchronize());
        boundary_kernel<VELOCITY_Y, 0><<<1, 1>>>();
    }
    checkCudaErrors(cudaDeviceSynchronize());
}
__device__ unsigned int rgba32(unsigned int r, unsigned int g, unsigned int b, unsigned int a) {
    /**
     * rgb 3 components to 32 bits rgb color [for texture]
     */
    r = clamp(r, 0u, 255u);
    g = clamp(g, 0u, 255u);
    b = clamp(b, 0u, 255u);
    a = clamp(a, 0u, 255u);
    return (a << 24) | (b << 16) | (g << 8) | r;
}

__device__ unsigned int rgba(float r, float g, float b, float a) {
    /**
     * rgb 0-1 to rgb 0-255 [for texture]
     */
    return rgba32(r*256, g*256, b*256, a*256);
}

__global__ void convertToImage(cudaSurfaceObject_t image) {
    /** converts the rgb and density into texture to visualize **/
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < NX && j < NY) {
        unsigned int index = i + j * NX;
        float r, g, b, a;

        a = density[index];
        r = red[index];
        g = green[index];
        b = blue[index];

        unsigned int pixel = rgba(r, g, b, a);
        surf2Dwrite(pixel, image, i * sizeof(pixel), j, cudaBoundaryModeClamp);
    }
}

void makeText(cudaSurfaceObject_t image) {
    /** converts the rgb and density into texture to visualize **/

    convertToImage<<<dimGrid, dimBlock>>>(image);
    checkCudaErrors(cudaDeviceSynchronize());
}
void fluidSimulationStep(float timestep, unsigned int iter,  int integration, int solver,
                         float jitter, float visc, float speedX, float speedY) {

    /**
     * ALGORITHM
     * advect fluid
     * diffuse
     * apply forces
     * pressure projection
     * advect free surface
     */
    timestep /= 100; // the GPU is too fast I had to reduce the timestep further
    advect(VELOCITY_X,  integration, timestep);
    advect(VELOCITY_Y,  integration, timestep);

    if(visc > 0.0001f) { // don't waste resource if viscosity is zero
        diffuse(VELOCITY_X, visc, iter, solver, timestep);
        diffuse(VELOCITY_Y, visc, iter, solver, timestep);
    }



    force<<<dimGrid, dimBlock>>>(jitter, speedX, speedY, timestep);
    cudaDeviceSynchronize();

    computePressure(solver, iter);

    advect(DENSITY,  integration, timestep);
    advect(RED,  integration, timestep);
    advect(GREEN,  integration, timestep);
    advect(BLUE,  integration, timestep);

    if(visc > 0.0001f) { // don't waste resource if viscosity is zero
        diffuse(DENSITY, visc, iter, solver, timestep);
        diffuse(RED, visc, iter, solver, timestep);
        diffuse(GREEN, visc, iter, solver, timestep);
        diffuse(BLUE, visc, iter, solver, timestep);
    }

    setBoundary(PRESSURE);
    setBoundary(VELOCITY_X);
    setBoundary(VELOCITY_Y);
}