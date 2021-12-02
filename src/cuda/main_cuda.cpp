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
#include <vector>

// openGL
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// imgui
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

// my code
#include "fluid.cuh"
#include "../shared/vboquad.h"
#include "../shared/glslprogram.h"


// window size
int gWindowWidth = 800;
int gWindowHeight = 500;

// fluid simulation
const int sizex = gWindowWidth; // grid width
const int sizey = gWindowHeight; // grid height
float timestep = 0.01f; // timestep
int iter = 5; // iterations for pressure/viscosity solve
int solver = 0; // default is jacobi
int integration = 0; // default is euler
int radius = 10; // how big is the density added
float densityVal = 5; // value of the density added
float viscosity = 0; // Default is no viscosity
float gpuTime = 0; // calculate time to execute one step in GPU
bool stopSimulation = false; // freeze simulation
bool randColor = true; // use one color for all or random color per click
float jitter = 50; // use for max random jitter in velocity
float speedX = 0; // velocity - X dir
float speedY = 5; // velocity - Y dir
glm::vec4 color(114.f/255, 169.f/255, 143.f/255, 1); // default color
glm::vec4 background(0); // background color

// graphics
GLSLProgram shaderProgram; // Shader
GLuint texture; // initialize texture in openGL
cudaGraphicsResource_t viewCudaResource; // initialize texture in cuda
Quad *quad; // initialize plane

float r=-1, g=-1, b=-1; // random color
int dragTime = 0; // for changing the random color after X drags
// colors to choose from
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


// query GPU functionality needed for CUDA, return false when not available
bool queryGPUCapabilitiesCUDA()
{
    // Device Count
    int devCount;

    // Get the Device Count
    checkCudaErrors(cudaGetDeviceCount(&devCount));

    // Print Device Count
    printf("Device(s): %i\n", devCount);

    for (int i = 0; i < devCount; i++) {
        cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, i));
        printf("Device #%d\n", i);
        printf("\tname: %s\n", prop.name);
        printf("\tmulti-processor count: %d\n",
               prop.multiProcessorCount);
        printf("\ttotal global memory: %zu\n",
               prop.totalGlobalMem);
        printf("\tshared memory per block: %zu\n",
               prop.sharedMemPerBlock);
        printf("\tnum registers per block: %d\n",
               prop.regsPerBlock);
        printf("\twarp size (in threads): %d\n",
               prop.warpSize);
        printf("\tmax threads per block: %d\n",
               prop.maxThreadsPerBlock);
        printf("\tmax grid size: %d\n",
               *prop.maxGridSize);
    }
    return devCount > 0;
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
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sizex, sizey, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    // connect with cuda texture
    checkCudaErrors(cudaGraphicsGLRegisterImage(&viewCudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

// one time initialization to setup the scene
void setupScene()
{
    // GLSLProgram
    shaderProgram.compileShader("../shaders/shader.vs");
    shaderProgram.compileShader("../shaders/shader.fs");
    shaderProgram.link();
    shaderProgram.use();

    // init objects in the scene
    quad = new Quad();

    // initialize texture
    generateTexBuffer();

    // initial fluid simulation
    fluidSimulationInitialize(sizex, sizey);
}

// render a frame (fluid simulation step + move results to texture)
void renderFrame(){

    // pass uniforms to GLSL
    shaderProgram.setUniform("color", color);
    shaderProgram.setUniform("randColor", randColor);
    shaderProgram.setUniform("background", background);

    // upload fluid simulation results to texture
    // cuda code are all boiler plates from here
    // https://stackoverflow.com/questions/20762828/crash-with-cuda-ogl-interop
    checkCudaErrors(cudaGraphicsMapResources(1, &viewCudaResource));
    cudaArray_t viewCudaArray;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewCudaResource, 0, 0));
    cudaResourceDesc viewCudaArrayResourceDesc;
    memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
    viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
    viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
    cudaSurfaceObject_t viewCudaSurfaceObject;
    checkCudaErrors(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));
    // fluid simulation step & time calculation code
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    /** fluid simulation step **/
    if(!stopSimulation) {
        fluidSimulationStep(timestep, iter, integration, solver, jitter, viscosity, speedX, speedY);
    }
    /** place results in the texture **/
    makeText(viewCudaSurfaceObject);
    // stop timer
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&gpuTime, start, stop));

    checkCudaErrors(cudaDestroySurfaceObject(viewCudaSurfaceObject));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &viewCudaResource));
    checkCudaErrors(cudaStreamSynchronize(0));

    // bind texture
    glBindTexture(GL_TEXTURE_2D, texture);
    /** the boiler plate above is intead of this
     * glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, sizex, sizey, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data);
     */

    // render plane
    quad->render();
    // release
    glBindTexture(GL_TEXTURE_2D, 0);
}

// user interface using imgui library
void imgui() {
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // list data
    static const char * integrations[]{"Euler", "RK-2", "RK-4"};
    static const char * solvers[]{"Jacobi", "Gauss-Seidel"};

    // set in top left
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    {
        ImGui::Begin("Main Configuration");
        if(ImGui::Button(!randColor? "Random Color": "Constant Color")) {
            randColor = !randColor;
        }
        if(!randColor) ImGui::ColorEdit3("Color", (float *) &color);
        ImGui::ColorEdit3("Background", (float *) &background);

        ImGui::Text("Numerical Configurations");
        ImGui::Combo("Integration Method", &integration, integrations, IM_ARRAYSIZE(integrations));
        ImGui::SliderFloat("Time Step (dt)", &timestep, 0, 0.1);
        ImGui::Combo("Solver", &solver, solvers, IM_ARRAYSIZE(solvers));
        ImGui::SliderInt("Iterations", &iter, 1, 20);

        ImGui::Text("Fluid");
        ImGui::SliderInt("Radius", &radius, 0, 50);
        ImGui::SliderFloat("Density", &densityVal, 0, 1000);
        ImGui::SliderFloat("Jitter", &jitter, 0, 500);
        ImGui::SliderFloat("Viscosity", &viscosity, 0, 100);
        ImGui::SliderFloat("X-Velocity", &speedX, -20, 20);
        ImGui::SliderFloat("Y-Velocity", &speedY, -20, 20);
        ImGui::Text("GPU Time %.2f ms", gpuTime);
        if(ImGui::Button("Reset")) {
            fluidSimulationReset();
        }
        ImGui::Checkbox("Stop? ", &stopSimulation);
        ImGui::End();
    }
}

// entry point
int main(int argc, char** argv)
{
    // query CUDA capabilities
    if (!queryGPUCapabilitiesCUDA())
    {
        // quit in case capabilities are insufficient
        exit(EXIT_FAILURE);
    }

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

    // make context current
    glfwMakeContextCurrent(window);

    // Callbacks
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    // get the frame buffer size
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // init the OpenGL API
    gladLoadGL();

    // ImGui initialization
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
        // poll and process input events
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
    // imgui destroy
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    // glfw destroy
    glfwTerminate();
    // fluid simulation destroy
    fluidSimulationDelete();
    return EXIT_SUCCESS;
}


// fix window height and width according to user
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    gWindowWidth = width;
    gWindowHeight = height;
    glViewport(0, 0, width, height);
}

// generate random color by choosing a color from the list above
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
        int mouse_x = xpos * sizex / (float) gWindowWidth;
        int mouse_y = ypos * sizey / (float) gWindowHeight;
        generateRandColor();

        // add density where the user clicked
        seed(mouse_x, mouse_y, radius, densityVal, r, g, b);
    }
}

// add density with user drag
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
    int mouse_x = xpos * sizex / (float)gWindowWidth;
    int mouse_y = ypos * sizey / (float)gWindowHeight;
    // only add ink inside box
    if(mouse_x > 0 && mouse_x < sizex - 2 && mouse_y > 0 && mouse_y < sizey - 2) {
        // change color when the user drags long enough
        if (r < 0 || g < 0 || b < 0 || dragTime > 10) {
            generateRandColor();
            dragTime = 0;
        }
        // add density where the user clicked
        // and move vector field where mouse is moving
        seed(mouse_x, mouse_y, radius, densityVal, r, g, b);
        generateRandColor();
        dragTime++;
    }
}