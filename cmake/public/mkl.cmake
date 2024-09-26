find_package(MKL QUIET)

if(TARGET caffe2::mkl)
  return()
endif()

add_library(caffe2::mkl INTERFACE IMPORTED)
target_include_directories(caffe2::mkl INTERFACE ${MKL_INCLUDE_DIR})
target_link_libraries(caffe2::mkl INTERFACE ${ONEMKL_LIBRARIES})
foreach(ONEMKL_LIB IN LISTS ONEMKL_LIBRARIES)
  if(EXISTS "${ONEMKL_LIB}")
    get_filename_component(ONEMKL_LINK_DIR "${ONEMKL_LIB}" DIRECTORY)
    if(IS_DIRECTORY "${ONEMKL_LINK_DIR}")
      target_link_directories(caffe2::mkl INTERFACE "${ONEMKL_LINK_DIR}")
    endif()
  endif()
endforeach()

# TODO: This is a hack, it will not pick up architecture dependent
# MKL libraries correctly; see https://github.com/pytorch/pytorch/issues/73008
set_property(
  TARGET caffe2::mkl PROPERTY INTERFACE_LINK_DIRECTORIES
  ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64 ${MKL_ROOT}/lib/intel64_win ${MKL_ROOT}/lib/win-x64)

if(UNIX)
  if(USE_STATIC_MKL)
    foreach(ONEMKL_LIB IN LISTS ONEMKL_LIBRARIES)
      if(NOT EXISTS "${ONEMKL_LIB}")
        continue()
      endif()

      get_filename_component(ONEMKL_LIB_NAME "${ONEMKL_LIB}" NAME)

      # Match archive libraries starting with "libmkl_"
      if(ONEMKL_LIB_NAME MATCHES "^libmkl_" AND ONEMKL_LIB_NAME MATCHES ".a$")
        target_link_options(caffe2::mkl INTERFACE "-Wl,--exclude-libs,${ONEMKL_LIB_NAME}")
      endif()
    endforeach()
  endif()
endif()
