//
// Created by Reem on 17/11/2021.
//
#pragma once
#pragma once
#include <cmath>
#include <cstdio>
#include "Grid.h"

Grid::Grid(unsigned int nx, unsigned int ny) :
        nx(nx), ny(ny), pixels(nx, ny),
        pressure(nx, ny), density(nx, ny),
        temp(nx, ny), divergence(nx, ny),
        velocity_x(nx, ny), velocity_y(nx, ny),
        red(nx, ny), green(nx, ny), blue(nx, ny)  {}

void Grid::reset() {
    /**
     * reset arrays to zero
     */
    pressure.reset();
    density.reset();
    velocity_y.reset();
    velocity_x.reset();
    temp.reset();
    divergence.reset();
    red.reset();
    green.reset();
    blue.reset();
}


void Grid::colorUpdate(Cell<float> &color, int i, int j, float c, float d) {
    /** update a color given parameters **/
    color(i, j) += density(i, j) * c;
}
void Grid::halfCircle(int i, int j, float distance, float power, float r, float g, float b) {
    /** update colors and density variables,
     * made into a function because I have two halves of a circle to update **/
    density(i, j) += power / (distance + 1);
    colorUpdate(red, i, j, r, distance);
    colorUpdate(green, i, j, g, distance);
    colorUpdate(blue, i, j, b, distance);
}
void Grid::seedColor(int x, int y, int radius, float power, float r, float g, float b) {
    /** given x, y position,
     * add density in the shape of the circle
     * and proportional to the distance from the radius and the given power
     */

    float distance;
    for (int j = y-radius; j <= y+radius; j++) {
        for (int i = x, k=x+1; (i-x)*(i-x)+(j-y)*(j-y) <= radius * radius; i--, k++) {
            if(j > 0 && j < ny - 2) {
                distance = sqrtf((i-x)*(i-x) + (j-y)*(j-y));
                if(i > 0 && i < nx - 2) { // left half of circle
                    halfCircle(i, j, distance, power, r, g, b);
                }
                if(k > 0 && k < nx - 2) { // right half of circle
                    halfCircle(k, j, distance, power, r, g, b);
                }
            }
        }
    }
}
void Grid::seed(int i, int j, float timestep, float dx, float dy, int radius, float power, float r, float g, float b) {
    seedColor(i, j, radius,  power, r, g, b);
}


float mix(float x, float y, float a) {
    /** opeGL mix function */
    return (1 - a) * x + a * y;
}
float linear(float y0, float y1, float x0, float x1, float x) {
    /** linear interpolation */
    // y = (y0 (x1 - x) + y1 (x - x0)) / (x1 - x0)
    // https://en.wikipedia.org/wiki/Linear_interpolation
    return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0);
}
float bilinear(float v11, float v21, float v12, float v22, float x, float x1, float x2, float y, float y1, float y2) {
    /** bilinear interpolation */
    // (Q11(x2-x)(y2-y)+Q21(x-x1)(y2-y)+Q12(x2-x)(y-y1)+Q22(x-x1)(y-y1)) / (x2-x1)(y2-y1)
    // https://en.wikipedia.org/wiki/Bilinear_interpolation
    return (v11 * (x2 - x ) * (y2 - y ) +
            v21 * (x  - x1) * (y2 - y ) +
            v12 * (x2 - x ) * (y  - y1) +
            v22 * (x  - x1) * (y  - y1))/ (x2 - x1) * (y2 - y1);
}
float Grid::interpolate(const Cell<float>& data, float x, float y) {
    /** Interpolation is done using:
     * bilinear interpolation if inside grid
     * linear interpolation if in the sides
     * nearest neighbor if outside the grid */

    int x1 = clamp((int)floorf(x), 0, (int)data.nx-1);
    int x2 = clamp((int)ceilf(x), 0, (int)data.nx-1);
    int y1 = clamp((int)floorf(y), 0, (int)data.ny-1);
    int y2 = clamp((int)ceilf(y), 0, (int)data.ny-1);

    // case 1: nearest neighbor
    if (x1 == x2 && y1 == y2)
        return data(x1, y1);

    // case 2: linear interpolation
    if (x1 == x2 && y1 != y2)
        return linear(data(x1, y1), data(x1, y2), y1, y2, y);
    if (y1 == y2 && x1 != x2)
        return linear(data(x1, y1), data(x2, y1), x1, x2, x);

    // case 3: bilinear interpolation
    return bilinear(data(x1, y1), data(x2, y1), data(x1, y2), data(x2, y2), x, x1, x2, y, y1, y2);
}

float Grid::computeGradient(const Cell<float>& data, int x, int y, int axis) {
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

    if(axis == 0){ // partial w.r.t. x
        if(x > 0 && x < (data.nx - 1)) // central difference
            return 0.5f * (data(x + 1, y) - data(x - 1, y));
        if(x == 0) // forward difference
            return data(x + 1, y) - data(x, y);
        else  // backward difference
            return data(x, y) - data(x - 1, y);
    } else { // partial w.r.t. y
        if(y > 0 && y < (data.ny - 1))  // central difference
            return 0.5f * (data(x, y + 1) - data(x, y - 1));
        if(y == 0) // forward difference
            return data(x, y + 1) - data(x, y);
        else // backward difference
            return data(x, y) - data(x, y - 1);
    }
}
float Grid::computeLaplacian(const Cell<float>& data,  int i, int j) {
    /** laplacian is computed as
     * 0.25 * (x(i+1, j) + x(i-1, j) + x(i, j+1) + x(i, j-1))
     **/
    float sum = 0;
    if (j > 0) {
        sum += data(i, j - 1);
    }
    if (i > 0) {
        sum += data(i - 1, j);
    }
    if (j < ny - 2) {
        sum += data(i, j + 1);
    }
    if (i < nx - 2) {
        sum += data(i + 1, j);
    }
    return sum;
}
void Grid::computeDivergence() {
    /**
     * divergence is  du/dx + dv/dy
     * negative divergence for pressure solve
     */
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            divergence(i, j) = - (computeGradient(velocity_x, i, j, 0) + computeGradient(velocity_y, i, j, 1));
        }
    }
}

void Grid::euler(Cell<float>& data_x, Cell<float>& data_y, float& timestep, int& i, int& j, float&x, float&y) {
    /** Euler forward integration to calculate the new position
     * x_{i+1} = x_{i} + v(x_{i}) * dt
     */
    x = (float)i - timestep * data_x(i, j);
    y = (float)j - timestep * data_y(i, j);
}
void Grid::rk2(Cell<float>& data_x, Cell<float>& data_y, float& timestep, int& i, int& j, float&x, float&y) {
    /** Runge-Kutta-2 integration to calculate the new position
     * x_{i+1} = x_{i} + v(x_{i} + v(x_{i}) * dt / 2) * dt
     */
    float u, v;
    x = (float)i - data_x(i, j) * timestep / 2;
    y = (float)j - data_y(i, j) * timestep / 2;
    u = interpolate(data_x, x, y);
    v = interpolate(data_y, x, y);
    x = (float)i - u * timestep;
    y = (float)j - v * timestep;
}
void Grid::rk4(Cell<float>& data_x, Cell<float>& data_y, float& timestep, int& i, int& j, float&x, float&y) {
    /** Runge-Kutta-4 integration to calculate the new position
     * x_{i+1} = x_{i} + (a + 2 b + 2 c + d) / 6
     * a = v(x_{i}) * dt
     * b = v(x_{i} + a/2) * dt
     * c = v(x_{i} + b/2) * dt
     * d = v(x_{i} + c) * dt
     */
    float a_x, a_y, b_x, b_y, c_x, c_y, d_x, d_y;
    a_x = interpolate(data_x, (float)i, (float)j) * timestep;
    a_y = interpolate(data_y, (float)i, (float)j) * timestep;

    b_x = interpolate(data_x, (float)i-a_x/2, (float)j-a_y/2) * timestep;
    b_y = interpolate(data_y, (float)i-a_x/2, (float)j-a_y/2) * timestep;

    c_x = interpolate(data_x, (float)i-b_x/2, (float)j-b_y/2) * timestep;
    c_y = interpolate(data_y, (float)i-b_x/2, (float)j-b_y/2) * timestep;

    d_x = interpolate(data_x, (float)i-c_x, (float)j-c_y) * timestep;
    d_y = interpolate(data_y, (float)i-c_x, (float)j-c_y) * timestep;

    x = (float)i - (a_x + 2 * b_x + 2 * c_x + d_x) / 6;
    y = (float)j - (a_y + 2 * b_y + 2 * c_y + d_y) / 6;
}

void Grid::jacobi(Cell<float>& x, Cell<float>& b, float alpha, float beta, unsigned int iter) {
    /** Jacobi iteration for solving systems of linear equations
     * Ax = b, A=D+L+U
     * x(k+1) = D^-1 (b - (L+U)x(k))
     * x_i(k+1) = (b_i - sum(j=1, j!=i, n){a_ij x_j(k)}) / a_ii
     * https://en.wikipedia.org/wiki/Jacobi_method
     * */
    for(int k = 0; k < iter; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                temp(i, j) = (alpha * computeLaplacian(x, i, j) + b(i, j)) * beta;
            }
        }
        x.swap(temp);
    }
}
void Grid::gaussSeidel(Cell<float>& x, Cell<float>& b, float alpha, float beta, unsigned int iter) {
    /** gauss-seidel iteration for solving systems of linear equations
     * Ax = b, A=LU
     * x(k+1) = L^-1 (b - Ux(k))
     * x_i(k+1) = (b_i - sum(j=1, i-1){a_ij x_j(k+1)} - sum(j=i+1, n){a_ij x_j(k)}) / a_ii
     * https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
     * */
    for(int k = 0; k < iter; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                x(i, j) = (alpha * computeLaplacian(x, i, j) + b(i, j)) * beta;
            }
        }
    }
}

void Grid::advect_euler(Cell<float>& quantity, float timestep) {
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
    float x, y;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            euler(velocity_x, velocity_y, timestep, i, j, x, y);
            temp(i, j) = interpolate(quantity, x, y);
        }
    }
    quantity.swap(temp);
}
void Grid::advect_rk2(Cell<float>& quantity, float timestep) {
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
    float x, y;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            rk2(velocity_x, velocity_y,timestep, i, j, x, y);
            temp(i, j) = interpolate(quantity, x, y);
        }
    }
    quantity.swap(temp);
}
void Grid::advect_rk4(Cell<float>& quantity, float timestep) {
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
    float x, y;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            rk4(velocity_x, velocity_y,timestep, i, j, x, y);
            temp(i, j) = interpolate(quantity, x, y);
        }
    }
    quantity.swap(temp);
}

void Grid::advect(Cell<float>& quantity, int integration, float timestep) {
    /**
     * Given some quantity Q on our simulation grid, how will Q change ∆t later?
     * broke down into 3 functions to make the if-statement run once per step
     * instead of running nx * ny times per step
    */

    if (integration == 1) { // RK2
        advect_rk2(quantity, timestep);
    }
    else if (integration == 2) { // RK4
        advect_rk4(quantity, timestep);
    }
    else { // EULER
        advect_euler(quantity, timestep);
    }
}

void Grid::diffuse(Cell<float> &quantity, float visc, unsigned int iter, int solver, float timestep) {
    /**
     * for solving viscosity term.
     * jacobi or gauss-seidel are used to solve the Poisson equation of viscosity
     */
    float a = timestep * visc;
    if(solver == 0)
        jacobi(quantity, quantity, a, 1.f/(1.f + 4.f * a), iter);
    else
        gaussSeidel(quantity, quantity, a, 1.f/(1.f + 4.f * a), iter);
}
void Grid::force(float timestep, float jitter, float speedX, float speedY) {
    /** apply force to the field
     * ie change velocity
     */
    float r;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            // random jittering around density
            r = rand() / (float)RAND_MAX;
            velocity_x(i, j) += mix(-jitter, jitter, r);
            r = rand() / (float)RAND_MAX;
            velocity_y(i, j) += mix(-jitter, jitter, r);

            velocity_x(i, j) += speedX * density(i, j);
            velocity_y(i, j) += speedY * density(i, j);

            // decay with time
//            velocity_x(i, j) *= 0.999f;
//            velocity_y(i, j) *= 0.999f;
//            density(i, j) *= 0.999f;

        }
    }

}
void Grid::subtractPressureGradient() {
    /** update velocity by subtracting grad pressure
     * based on Helmholtz-Hodge Decomposition Theorem
     */
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            velocity_x(i, j) -= computeGradient(pressure, i, j, 0);
            velocity_y(i, j) -= computeGradient(pressure, i, j, 1);
        }
    }
}
void Grid::computePressure(unsigned int iter, int solver) {
    /**
     * compute the divergence
     * jacobi/gauss-seidel to find grad pressure using pressure laplacian - divergence
     * update vector field by subtracting velocity from pressure
     */
    pressure.reset();
    computeDivergence();
    if(solver == 0)
        jacobi(pressure, divergence, 1, 0.25f, iter);
    else
        gaussSeidel(pressure, divergence, 1, 0.25f, iter);
    subtractPressureGradient();
}
void Grid::setBoundary(Cell<float> &quantity, int flip)
{
    /**
     * fluid can't escape the box
     * no-slip (zero) velocity boundary condition
     * pure Neumann pressure boundary condition
     * to satisfy the conditions, the boundary values should be:
     * same value as the neighboring cell for pressure
     * the negative of the neighboring cell value for velocity
     */
    for(int i = 1; i < nx - 1; i++) {
        quantity(i, 0) = flip * quantity(i, 1);
        quantity(i, ny-1) = flip * quantity(i, ny-2);
    }
    for(int j = 1; j < ny - 1; j++) {
        quantity(0  , j) = flip * quantity(1  , j);
        quantity(nx-1, j) = flip * quantity(nx-2, j);
    }

    quantity(0, 0) =  0.5f * (quantity(1, 0) + quantity(0, 1));
    quantity(0, ny-1) =  0.5f * (quantity(1, ny-1) + quantity(0, ny-2));
    quantity(nx-1, 0) =  0.5f * (quantity(nx-1, 1) + quantity(nx-2, 0));
    quantity(nx-1, ny-1) = 0.5f * (quantity(nx-2, ny-1) + quantity(nx-1, ny-2));
}


// rgb 3 components to 32 bits rgb color [for texture]
unsigned int rgba32(unsigned int r, unsigned int g, unsigned int b, unsigned int a){
    r = clamp(r, 0u, 255u);
    g = clamp(g, 0u, 255u);
    b = clamp(b, 0u, 255u);
    a = clamp(a, 0u, 255u);
    return (a << 24) | (b << 16) | (g << 8) | r;
}
// rgb 0-1 to rgb 0-255 [for texture]
unsigned int rgba(float r, float g, float b, float a){
    return rgba32(r*256, g*256, b*256, a*256);
}

void Grid::makeText() {
    /** converts the rgb and density into texture to visualize **/
    float r, g, b, a;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            a = density(i, j);
            r = red(i, j);
            g = green(i, j);
            b = blue(i, j);

            pixels(i, j) = rgba(r, g, b, a);
        }
    }

}
void Grid::step(float timestep, unsigned int iter,  int integration, int solver,
                float jitter, float visc, float speedX, float speedY) {

    /**
     * ALGORITHM
     * advect fluid
     * diffuse
     * apply forces
     * pressure projection
     * advect free surface
     */


    advect(velocity_x, integration, timestep);
    advect(velocity_y, integration, timestep);

    if(visc > 0.0001f) { // don't waste resource if viscosity is zero
        diffuse(velocity_x, visc, iter, solver, timestep);
        diffuse(velocity_y, visc, iter, solver, timestep);
    }

    force(timestep, jitter, speedX, speedY);
    computePressure(iter, solver);

    advect(density, integration, timestep);
    advect(red, integration, timestep);
    advect(green, integration, timestep);
    advect(blue, integration, timestep);
    if(visc > 0.0001f) { // don't waste resource if viscosity is zero
        diffuse(density, visc, iter, solver, timestep);
        diffuse(red, visc, iter, solver, timestep);
        diffuse(green, visc, iter, solver, timestep);
        diffuse(blue, visc, iter, solver, timestep);
    }
    setBoundary(pressure, 1);
    setBoundary(velocity_x, -1);
    setBoundary(velocity_y, -1);
}