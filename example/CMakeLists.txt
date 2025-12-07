include_directories(${CMAKE_SOURCE_DIR})

add_executable(silero_vad silero-vad.cc)
target_link_libraries(silero_vad PRIVATE onnx-executor)