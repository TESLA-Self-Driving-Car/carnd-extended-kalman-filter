project(ExtendedKF)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

file(GLOB SOURCES "src/*.cpp")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

include_directories(include)

include_directories(/usr/local/include)
include_directories(/usr/local/opt/openssl/include)

link_directories(/usr/local/lib)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1.11.0/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

add_library(fusion_ekf ${SOURCES})
target_include_directories (fusion_ekf PUBLIC include)

add_executable(ExtendedKF tests/main.cpp)
target_link_libraries(ExtendedKF fusion_ekf z ssl uv uWS)
