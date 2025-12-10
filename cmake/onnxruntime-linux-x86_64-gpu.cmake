if(NOT ENABLE_ONNX_EXECUTOR_GPU)
  message(FATAL_ERROR "This cmake script for NVIDIA GPU only. Should be built with -DENABLE_ONNX_EXECUTOR_GPU=ON")
endif()

find_package(CUDA REQUIRED)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA NOT FOUND, onnxruntime-gpu depends on CUDA")
  else()
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
  endif()

set(onnxruntime_URL 
  "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz"
)
set(onnxruntime_HASH "SHA256=2083e361072a79ce16a90dcd5f5cb3ab92574a82a3ce0ac01e5cfa3158176f53")

set(onnxruntime_possible_file_locations
  ${CMAKE_SOURCE_DIR}/third-part/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz
)

foreach(f IN LISTS onnxruntime_possible_file_locations)
  if(EXISTS ${f})
    set(onnxruntime_URL ${f})
    file(TO_CMAKE_PATH "${onnxruntime_URL}" onnxruntime_URL)
    message(STATUS "Found local downloaded onnxruntime: ${onnxruntime_URL}")
    break()
  endif()
endforeach()

FetchContent_Declare(onnxruntime
  URL
    ${onnxruntime_URL}
  URL_HASH          ${onnxruntime_HASH}
)

FetchContent_GetProperties(onnxruntime)
if(NOT onnxruntime_POPULATED)
  message(STATUS "Downloading onnxruntime from ${onnxruntime_URL}")
  FetchContent_Populate(onnxruntime)
endif()
message(STATUS "onnxruntime is downloaded to ${onnxruntime_SOURCE_DIR}")

find_library(location_onnxruntime onnxruntime
  PATHS
  "${onnxruntime_SOURCE_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)
message(STATUS "location_onnxruntime: ${location_onnxruntime}")
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${location_onnxruntime}
  INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
)

find_library(location_onnxruntime_cuda_lib onnxruntime_providers_cuda
  PATHS
  "${onnxruntime_SOURCE_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)
message(STATUS "location_onnxruntime_cuda_lib: ${location_onnxruntime_cuda_lib}")
add_library(onnxruntime_providers_cuda SHARED IMPORTED)
set_target_properties(onnxruntime_providers_cuda PROPERTIES
  IMPORTED_LOCATION ${location_onnxruntime_cuda_lib}
  IMPORTED_NO_SONAME TRUE
)

find_library(location_onnxruntime_providers_shared_lib onnxruntime_providers_shared
  PATHS
  "${onnxruntime_SOURCE_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)
message(STATUS "location_onnxruntime_providers_shared_lib: ${location_onnxruntime_providers_shared_lib}")
add_library(onnxruntime_providers_shared SHARED IMPORTED)
set_target_properties(onnxruntime_providers_shared PROPERTIES
  IMPORTED_LOCATION ${location_onnxruntime_providers_shared_lib}
)

find_library(location_onnxruntime_providers_tensorrt_lib onnxruntime_providers_tensorrt
  PATHS
  "${onnxruntime_SOURCE_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)
message(STATUS "location_onnxruntime_providers_tensorrt_lib: ${location_onnxruntime_providers_tensorrt_lib}")
add_library(onnxruntime_providers_tensorrt SHARED IMPORTED)
set_target_properties(onnxruntime_providers_tensorrt PROPERTIES
  IMPORTED_LOCATION ${location_onnxruntime_providers_tensorrt_lib}
)

file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime*")
file(COPY ${onnxruntime_lib_files} DESTINATION ${CMAKE_BINARY_DIR}/lib)
message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
install(FILES ${onnxruntime_lib_files} DESTINATION lib)