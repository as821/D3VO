ADD_LIBRARY(types_sim3 ${G2O_LIB_TYPE}
  types_seven_dof_expmap.cpp
  types_seven_dof_expmap.h
)


include_directories(${PROJECT_SOURCE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CHOLMOD_INCLUDE_DIR})
include_directories(${CSPARSE_INCLUDE_DIR})

# (astange) Have to include the "include" path to your Python installation to find the "Python.h" file needed for pybind11. 
# My setup is screwy, so may just be a local issue for me 
# https://cmake.org/cmake/help/latest/module/FindPython.html
find_package (Python COMPONENTS Interpreter Development)
include_directories(${Python_INCLUDE_DIRS}) 

# pybind11 (version 2.2.1)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/EXTERNAL/pybind11/tools)
include_directories(${PROJECT_SOURCE_DIR}/EXTERNAL/pybind11/include)
include(pybind11Tools)



SET_TARGET_PROPERTIES(types_sim3 PROPERTIES OUTPUT_NAME ${LIB_PREFIX}types_sim3)

TARGET_LINK_LIBRARIES(types_sim3 types_sba)

INSTALL(TARGETS types_sim3
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

FILE(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

INSTALL(FILES ${headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/g2o/types/sim3)

