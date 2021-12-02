/** FROM KAUST-GPU 380 course by Prof Markus Hadwiger
 * https://vccvisualization.org/CS380_GPU_and_GPGPU_Programming/
 **/
#include "vboquad.h"

#include "glad/glad.h" 

// include glfw library: http://www.glfw.org/
#include <GLFW/glfw3.h>

Quad::Quad()
{
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
            // positions         // texture coords
             1.f,  1.f, 0.0f,  1.0f, 1.0f, // top right
             1.f, -1.f, 0.0f,  1.0f, 0.0f, // bottom right
            -1.f, -1.f, 0.0f,  0.0f, 0.0f, // bottom left
            -1.f,  1.f, 0.0f,  0.0f, 1.0f  // top left
    };
    unsigned int indices[] = {
            0, 1, 3, // first triangle
            1, 2, 3  // second triangle
    };

    glGenVertexArrays( 1, &quadVAOHandle);
    glBindVertexArray(quadVAOHandle);

    unsigned int handle[2];
    glGenBuffers(2, handle);

    glBindBuffer(GL_ARRAY_BUFFER, handle[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handle[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

void Quad::render() {
    glBindVertexArray(quadVAOHandle);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, ((GLubyte *)NULL + (0)));
}
