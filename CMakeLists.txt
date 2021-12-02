cmake_minimum_required (VERSION 3.6 FATAL_ERROR)


project (fluid_sim LANGUAGES CUDA CXX C)
# Adding cuda libraries
find_package(CUDA)
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")


# Adding compile units to executable
CUDA_ADD_EXECUTABLE(${PROJECT_NAME}
        src/shared/glslprogram.cpp
        src/shared/vboquad.cpp

        # GPU implementation
        src/cuda/main_cuda.cpp
        src/cuda/fluid.cu

        # CPU implementation
#        src/cpu/main.cpp
#        src/cpu/Grid.cpp

        common/imgui/imconfig.h
        common/imgui/imgui_demo.cpp
        common/imgui/imgui_draw.cpp
        common/imgui/imgui_impl_glfw.cpp
        common/imgui/imgui_impl_glfw.h
        common/imgui/imgui_impl_opengl3.cpp
        common/imgui/imgui_internal.h
        common/imgui/imgui_widgets.cpp
        common/imgui/imgui.cpp
        common/imgui/imgui.h
        common/imgui/imstb_rectpack.h
        common/imgui/imstb_textedit.h
        common/imgui/imstb_truetype.h)


# Adding glad from common folder
include_directories(common)
add_library(glad common/glad/glad.c)
target_link_libraries(${PROJECT_NAME} glad)


# C++ Compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Setting up CUDA compiler (NVCC) flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
set(CUDA_HOST_COMPILER clang++)

# Adding cuda libraries
include_directories(${CUDA_INCLUDE_DIRS})
SET(ALL_CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})
SET(LIBS ${LIBS} ${ALL_CUDA_LIBS})
message(STATUS "CUDA_LIBRARIES: ${CUDA_INCLUDE_DIRS} ${ALL_CUDA_LIBS}")
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION OFF)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}  ${CUDA_LIBRARIES} cuda X11)

# Adding glfw3
add_subdirectory(/home/gin/glfw /home/gin/glfw/src)
target_link_libraries(${PROJECT_NAME}  glfw)
target_compile_features(${PROJECT_NAME}  PUBLIC cxx_std_11)