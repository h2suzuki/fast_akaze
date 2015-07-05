# CMakeLists.txt based on opencv/samples/cpp/example_cmake

# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(fast_akaze)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Needs C++11 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if (COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  message(STATUS "${CMAKE_CXX_COMPILER} has no C++11 support")
endif()


# Find threading support e.g. pthread
find_package(Threads)
if (THREADS_HAVE_PTHREAD_ARG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add akaze main code
add_subdirectory(akaze)

# Declare the executable target built from your sources
add_executable(test_fast_akaze main.cpp)

# Link your application with OpenCV libraries
target_link_libraries(test_fast_akaze akaze ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
