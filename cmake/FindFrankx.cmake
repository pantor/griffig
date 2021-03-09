if(NOT TARGET frankx)
  set(frankx_LIB_NAME "frankx")
  set(movex_LIB_NAME "movex")
  set(frankx_LIB_DIR "/home/berscheid/Documents/frankx/build")
  set(frankx_INCLUDE_DIR "/home/berscheid/Documents/frankx/include")

  find_library(frankx_LIB ${frankx_LIB_NAME} PATHS ${frankx_LIB_DIR})
  find_library(movex_LIB ${movex_LIB_NAME} PATHS ${frankx_LIB_DIR})

  add_library(frankx::frankx INTERFACE IMPORTED)
  target_include_directories(frankx::frankx INTERFACE ${frankx_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(frankx::frankx INTERFACE ${frankx_LIB})

  add_library(frankx::movex INTERFACE IMPORTED)
  target_include_directories(frankx::movex INTERFACE ${frankx_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(frankx::movex INTERFACE ${movex_LIB})
endif()
