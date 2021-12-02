//
// Created by Reem on 24/11/2021.
//
#pragma once
#ifndef FLUID_SIM_CELL_H
#define FLUID_SIM_CELL_H

template <typename T>
T clamp(T x, T min_val, T max_val) {
    return  x < min_val ? min_val : x > max_val ? max_val : x;
}

template <typename T>
struct Cell {
    unsigned int nx, ny, size;
    T* data;

    // Constructors
    Cell(unsigned int sizex, unsigned int sizey) :
            nx(sizex), ny(sizey)
    {
        // initialize to an empty 2D array
        size = nx * ny;
        data = new T[size];
        memset(data, 0, size * sizeof (float));
    }
    ~Cell() {
        delete[] data;
    }

    // accessors
    unsigned int id(unsigned int i, unsigned int j) const {
        return i + j * nx;
    }

    T& operator () (unsigned int i, unsigned int j) {
        return data[id(i, j)];
    }
    T& operator () (unsigned int i, unsigned int j) const {
        return data[id(i, j)];
    }
    
    // helpers
    void swap(Cell<T> &other){
        std::swap(data, other.data);
    }
    // reset variables
    void reset() {
        memset(data, 0, size * sizeof (float));
    }
};


#endif //FLUID_SIM_CELL_H
