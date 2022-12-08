ADD_LIBRARY(types_sba ${G2O_LIB_TYPE}
  types_sba.h     types_six_dof_expmap.h
  types_sba.cpp   types_six_dof_expmap.cpp
  g2o_types_sba_api.h
)


include_directories(${PROJECT_SOURCE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CHOLMOD_INCLUDE_DIR})
include_directories(${CSPARSE_INCLUDE_DIR})

# (astange) Have to include the path to your Python installation to find the "Python.h" file needed by pybind11. 
# This may just be a local issue for me, but this resolves it 
# https://cmake.org/cmake/help/latest/module/FindPython.html
find_package (Python COMPONENTS Interpreter Development)
include_directories(${Python_INCLUDE_DIRS}) 

# pybind11 (version 2.2.1)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/EXTERNAL/pybind11/tools)
include_directories(${PROJECT_SOURCE_DIR}/EXTERNAL/pybind11/include)
include(pybind11Tools)


SET_TARGET_PROPERTIES(types_sba PROPERTIES OUTPUT_NAME ${LIB_PREFIX}types_sba)

TARGET_LINK_LIBRARIES(types_sba core types_slam3d)

INSTALL(TARGETS types_sba
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

FILE(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

INSTALL(FILES ${headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/g2o/types/sba)
