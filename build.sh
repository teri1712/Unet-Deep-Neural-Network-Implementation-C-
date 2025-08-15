#!/bin/bash

# Create a build directory if it doesn't exist
mkdir -p build
cd build

# Run CMake and build the project
cmake -DOpenCV_DIR=$1 ..
cmake --build . --config Release
