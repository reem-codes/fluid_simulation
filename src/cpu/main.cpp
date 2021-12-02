//
// Created by Reem on 20/11/2021.
//

// includes
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>
#include <cassert>
#include <cmath>
#include <vector>

// graphics
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// imgui
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

// cuda
#include <cuda_runtime.h>

// my code
#include "Grid.h"
#include "../shared/vboquad.h"
#include "../shared/glslprogram.h"

// window size
int gWindowWidth = 512;
int gWindowHeight = 512;
int last_mouse_x=-1, last_mouse_y=-1;

// fluid simulation
const int nx = 256; // grid width
const int ny = 256; // grid height
float timestep = 0.02f; // timestep
int iter = 5; // iterations for pressure solve
int solver = 0; // default is jacobi
int integration = 0; // default is euler
int radius = 10; // how big is the density added
float density = 5; // value of the density added
float viscosity = 0; // Default is no viscosity
float cpuTime = 0; // calculate time to execute one step in CPU
bool stopSimulation = false; // freeze simulation
bool randColor = true; // use one color for all or random color per click
float jitter = 20; // use for max random jitter in velocity
float speedX = 0; // velocity factor of fluid - X dir
float speedY = 5; // velocity factor of fluid - Y dir
glm::vec4 color(114.f/255, 169.f/255, 143.f/255, 1); // default color
glm::vec4 background(0); // background color

Grid fluidGrid(nx, ny); // initialize grid

// graphics
GLSLProgram shaderProgram; // Shader
GLuint texture; // initialize texture
Quad *quad; // initialize plane
float r=-1, g=-1, b=-1; // random color
int dragTime = 0;
char* colors[] = {"ABEBD2", "DB222A", "F3FFBD", "70C1B3", "247BA0", "B2DBBF", "FF1654", "EDB458", "E8871E", "BAD4AA", "BF8B85", "DABECA", "5D5F71", "AF1B3F", "DF9B6D", "EFC69B", "90BEDE", "68EDC6", "F3B391", "FEC601", "EA7317", "2364AA", "3DA5D9", "73BFB8", "FFF9A5", "AFC2D5", "CCDDD3", "B48B7D", "DFEFCA", "C879FF", "F865B0", "A53F2B", "F75C03", "2274A5", "D90368", "00CC66", "E2B4BD", "FBCAEF", "9B5DE5", "823329", "A26769", "EF3054", "E637BF", "93A8AC", "00F5D4", "826AED", "FFB7FF", "F15BB5", "FEE440", "CAFF8A", "DECDF5", "3BF4FB", "1B998B", "BFCC94", "E6AACE", "998DA0", "F4D35E", "E4FF1A", "B9C0DA", "C4E7D4", "F95738", "EE964B", "6EEB83", "FF5714", "1BE7FF"};
// callbacks
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);

// glfw error callback
void glfwErrorCallback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

// OpenGL error debugging callback
void APIENTRY glDebugOutput(GLenum source,
                            GLenum type,
                            GLuint id,
                            GLenum severity,
                            GLsizei length,
                            const GLchar *message,
                            const void *userParam)
{
    // ignore non-significant error/warning codes
    if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

    std::cout << "---------------" << std::endl;
    std::cout << "Debug message (" << id << "): " << message << std::endl;

    switch (source)
    {
        case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
        case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
        case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
    } std::cout << std::endl;

    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
        case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
        case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
        case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
        case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
        case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
    } std::cout << std::endl;

    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
        case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
    } std::cout << std::endl;
    std::cout << std::endl;
}


// init application
bool initApplication(int argc, char **argv)
{

    std::string version((const char *)glGetString(GL_VERSION));
    std::stringstream stream(version);
    unsigned major, minor;
    char dot;

    stream >> major >> dot >> minor;

    assert(dot == '.');
    if (major > 3 || (major == 2 && minor >= 0)) {
        std::cout << "OpenGL Version " << major << "." << minor << std::endl;
    } else {
        std::cout << "The minimum required OpenGL version is not supported on this machine. Supported is only " << major << "." << minor << std::endl;
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, gWindowWidth, gWindowHeight);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    return true;
}

// initialize texture, to be filled with fluids later
void generateTexBuffer() {

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, nx, ny, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

// one time initialization to setup the scene
void setupScene()
{
    // Use GLSLProgram class or your own solution
    shaderProgram.compileShader("../shaders/shader.vs");
    shaderProgram.compileShader("../shaders/shader.fs");
    shaderProgram.link();
    shaderProgram.use();

    // init objects in the scene
    quad = new Quad();

    // initialize texture
    generateTexBuffer();
}

// render a frame (fluid simulation step + move results to texture)
void renderFrame(){

    // graphics
    shaderProgram.setUniform("color", color);
    shaderProgram.setUniform("randColor", randColor);
    shaderProgram.setUniform("background", background);
    // fluid simulation step & time calculation code
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if(!stopSimulation) {
        fluidGrid.step(timestep, iter, integration, solver, jitter, viscosity, speedX, speedY);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpuTime, start, stop);
    // upload pixels to texture
    fluidGrid.makeText();
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGBA, GL_UNSIGNED_BYTE, fluidGrid.pixels.data);

    // render plane
    quad->render();
}

// user interface using imgui library
void imgui() {
// Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    static const char * integrations[]{"Euler", "RK-2", "RK-4"};
    static const char * solvers[]{"Jacobi", "Gauss-Seidel"};

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    {

        ImGui::Begin("Main Configuration");

        if(ImGui::Button(!randColor? "Random Color":"Constant Color")) {
            randColor = !randColor;
        }
        if(!randColor) ImGui::ColorEdit3("Color", (float *) &color);
        ImGui::ColorEdit3("Background", (float *) &background);

        ImGui::Text("Numerical Configurations");
        ImGui::Combo("Integration Method", &integration, integrations, IM_ARRAYSIZE(integrations));
        ImGui::SliderFloat("Time Step (dt)", &timestep, 0, .1);
        ImGui::Combo("Solver", &solver, solvers, IM_ARRAYSIZE(solvers));
        ImGui::SliderInt("Iterations", &iter, 1, 20);

        ImGui::Text("Fluid");
        ImGui::SliderInt("Radius", &radius, 0, 50);
        ImGui::SliderFloat("Density", &density, 0, 1000);
        ImGui::SliderFloat("Jitter", &jitter, 0, 500);
        ImGui::SliderFloat("Viscosity", &viscosity, 0, 100);
        ImGui::SliderFloat("Velocity X", &speedX, -20, 20);
        ImGui::SliderFloat("Velocity Y", &speedY, -20, 20);
        ImGui::Text("CPU Time %.2f ms", cpuTime);
        if(ImGui::Button("Reset")) {
            fluidGrid.reset();
        }
        ImGui::Checkbox("Stop? ", &stopSimulation);
        ImGui::End();
    }
}

// entry point
int main(int argc, char** argv)
{

    // set glfw error callback
    glfwSetErrorCallback(glfwErrorCallback);

    // init glfw
    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    // init glfw window
    GLFWwindow* window;
    window = glfwCreateWindow(gWindowWidth, gWindowHeight, "Fluid Simulation - Reem :D", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }


    // make context current (once is sufficient)
    glfwMakeContextCurrent(window);

    // Callbacks
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    // get the frame buffer size
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // init the OpenGL API (we need to do this once before any calls to the OpenGL API)
    gladLoadGL();

    // --- Dear ImGui

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");


    // init application
    if (!initApplication(argc, argv)) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // setting up scene
    setupScene();

    // start the main loop
    // loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {
        // poll and process input events (keyboard, mouse, window, ...)
        glfwPollEvents();

        // imgui code for interface
        imgui();
        // render one frame
        renderFrame();


        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // swap front and back buffers
        glfwSwapBuffers(window);


    }
    // terminate
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return EXIT_SUCCESS;
}


// fix window height and width according to user
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    gWindowWidth = width;
    gWindowHeight = height;
    glViewport(0, 0, width, height);
}

void generateRandColor() {
    int id = mix(0, 64, (float)rand()/(float)RAND_MAX);
    int ir, ig, ib;
    sscanf(colors[id], "%02x%02x%02x", &ir, &ig, &ib);
    r = (float)ir / 255.f;
    g = (float)ig / 255.f;
    b = (float)ib / 255.f;
    printf("COLOR: (%f, %f, %f)\n", r, g, b);
}

// add density where the user clicked
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse){
        return;
    }
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        double xpos, ypos;
        //getting cursor position
        glfwGetCursorPos(window, &xpos, &ypos);

        // from mouse coord to screen coord
        ypos = gWindowHeight - 1 - ypos;
        int mouse_x = xpos * nx / (float) gWindowWidth;
        int mouse_y = ypos * ny / (float) gWindowHeight;
        generateRandColor();

        // add density where the user clicked
        fluidGrid.seedColor(mouse_x, mouse_y, radius, density, r, g, b);
    }
}
// move velocity with user drag
static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse){
        return;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
    {
        return;
    }

    // from mouse coord to screen coord
    ypos = gWindowHeight - 1 - ypos;
    int mouse_x = xpos * nx / (float)gWindowWidth;
    int mouse_y = ypos * ny / (float)gWindowHeight;
    if(mouse_x > 0 && mouse_x < nx - 2 && mouse_y > 0 && mouse_y < ny - 2) {
        if (last_mouse_x == -1 || last_mouse_y == -1) {
            last_mouse_x = mouse_x;
            last_mouse_y = mouse_y;
        }
        if (r < 0 || g < 0 || b < 0 || dragTime > 10) {
            generateRandColor();
            dragTime = 0;
        }
        // add density where the user clicked
        // and move vector field where mouse is moving
        fluidGrid.seed(mouse_x, mouse_y, timestep, (float)(mouse_x-last_mouse_x)/nx, (float)(mouse_y-last_mouse_y)/ny, radius, density, r, g, b);
        generateRandColor();
        dragTime++;
    }
}
