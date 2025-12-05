function(download_onnxruntime)
  include(FetchContent)
  set(ONNX_VERSION "1.14.0")

  message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
  message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64)
      include(onnxruntime-linux-x86_64)
    else()
      #TODO(lianghu): support other platform
    endif()
  elseif(WIN32)
    # TODO(lianghu)
  else()
    message(FATAL_ERROR "Only support Linux and Windows.")
  endif()
endfunction()

download_onnxruntime()
