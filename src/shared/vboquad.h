#ifndef Quad_H
#define Quad_H

/** FROM KAUST-GPU 380 course by Prof Markus Hadwiger
 * https://vccvisualization.org/CS380_GPU_and_GPGPU_Programming/
 **/
class Quad
{

private:
    unsigned int quadVAOHandle;

public:
    Quad();

    void render();
};

#endif // Quad_H
