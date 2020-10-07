cmake_minimum_required(VERSION 3.5)

project(LocalExpStereo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/maxflow maxflow_src)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/LocalExpansionStereo LocalExpansionStereo_src)

include_directories(${CMAKE_CURRENT_LIST_DIR}/maxflow)
include_directories(${CMAKE_CURRENT_LIST_DIR}/LocalExpansionStereo)

add_executable(LocalExpStereoExample ${CMAKE_CURRENT_LIST_DIR}/example/main.cpp ${LocalExpansionStereo_src})
add_executable(LocalExpStereoCamera ${CMAKE_CURRENT_LIST_DIR}/example/stereo.cpp ${LocalExpansionStereo_src})
add_library(libmaxflow STATIC ${maxflow_src})

target_link_libraries(LocalExpStereoExample libmaxflow)
target_link_libraries(LocalExpStereoExample ${OpenCV_LIBS})

target_link_libraries(LocalExpStereoCamera libmaxflow)
target_link_libraries(LocalExpStereoCamera ${OpenCV_LIBS})
