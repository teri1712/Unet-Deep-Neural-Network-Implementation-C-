cmake_minimum_required(VERSION 3.18)
project(Unet)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_PREFIX_PATH ./libtorch)

find_package (Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(SOURCE_FILES source.cpp)

add_executable(TrainModel TrainModel.cpp ${SOURCE_FILES})
add_executable(TestModel TestModel.cpp ${SOURCE_FILES})

target_link_libraries(TrainModel ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(TestModel ${TORCH_LIBRARIES} ${OpenCV_LIBS})

set_property (TARGET TrainModel PROPERTY CXX_STANDARD 17)
set_property (TARGET TestModel PROPERTY CXX_STANDARD 17)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET TrainModel
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:TrainModel>)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET TestModel
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:TestModel>)
endif (MSVC)
