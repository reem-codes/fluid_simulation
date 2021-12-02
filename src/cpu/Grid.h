//
// Created by Reem on 17/11/2021.
//

#ifndef FLUID_SIM_GRID_H
#define FLUID_SIM_GRID_H


#include <cstring>
#include <cmath>
#include <algorithm>
#include "Cell.h"

// functions used by simulation and by main script
float mix(float x, float y, float a);
float linear(float y0, float y1, float x0, float x1, float x);
float bilinear(float v11, float v21, float v12, float v22, float x, float x1, float x2, float y, float y1, float y2);

struct Grid {
    // Data
    unsigned int nx, ny;
    Cell<float> velocity_x, velocity_y, pressure, density, divergence, temp, red, green, blue;
    Cell<unsigned int> pixels;   // initialize texture data


    // Constructors
    Grid(unsigned int nx, unsigned int ny);

    // Helpers
    void reset();
    float interpolate(const Cell<float> &data, float x, float y);

    void colorUpdate(Cell<float> &color, int i, int j, float c, float d);
    void halfCircle(int i, int j, float distance, float power, float r, float g, float b);
    void seedColor(int x, int y, int radius, float power, float r, float g, float b);
    void seed(int i, int j, float timestep, float dx, float dy, int radius, float power, float r, float g, float b);

    // math
    float computeGradient(const Cell<float>& data, int x, int y, int axis);
    float computeLaplacian(const Cell<float>& data, int i, int j) ;

    void euler(Cell<float>& data_x, Cell<float>& data_y, float &timestep, int &i, int &j, float &x, float &y);
    void rk2(Cell<float>& data_x, Cell<float>& data_y, float &timestep, int &i, int &j, float &x, float &y);
    void rk4(Cell<float>& data_x, Cell<float>& data_y, float &timestep, int &i, int &j, float &x, float &y);

    void jacobi(Cell<float>& x, Cell<float>& b, float alpha, float beta, unsigned int iter);
    void gaussSeidel(Cell<float>& x, Cell<float>& b, float alpha, float beta, unsigned int iter);

    // Solvers
    void advect_rk2(Cell<float>& quantity,float timestep);
    void advect_rk4(Cell<float>& quantity, float timestep);
    void advect_euler(Cell<float>& quantity, float timestep);
    void advect(Cell<float>& quantity, int integration, float timestep);
    void diffuse(Cell<float> &quantity, float visc, unsigned int iter, int solver, float timestep);
    void force(float timestep, float jitter, float speedX, float speedY);
    void computeDivergence();
    void subtractPressureGradient();
    void computePressure(unsigned int iter, int solver);
    void setBoundary(Cell<float> &quantity, int flip);
    void step(float timestep, unsigned int iter,  int integration, int solver,
              float jitter, float visc, float speedX, float speedY);
    void makeText();
};

#endif //FLUID_SIM_GRID_H
