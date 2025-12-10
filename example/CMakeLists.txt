include_directories(${CMAKE_SOURCE_DIR})

find_package(Threads REQUIRED)

add_executable(silero_vad silero-vad.cc)
target_link_libraries(silero_vad PRIVATE onnx-executor)

add_executable(onnxruntime-cuda-example onnxruntime-cuda-example.cc)
target_link_libraries(onnxruntime-cuda-example PRIVATE onnx-executor)

add_executable(onnxruntime-trt-example onnxruntime-trt-example.cc)
target_link_libraries(onnxruntime-trt-example PRIVATE onnx-executor)

add_executable(multi_thread_load multi_thread_load.cc)
target_link_libraries(multi_thread_load PRIVATE onnx-executor Threads::Threads)