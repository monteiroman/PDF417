cmake_minimum_required(VERSION 2.8)

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})
MESSAGE(STATUS "Nombre del módulo: ${ProjectId}")
project( ${ProjectId} )

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_BUILD_TYPE "Release")

aux_source_directory(./src SRC_LIST)
include_directories(./hpp SRC_LIST)
FILE(GLOB_RECURSE LibFiles "./headers/*.h")
add_custom_target(headers SOURCES ${LibFiles})

add_executable( ${PROJECT_NAME} ${SRC_LIST} )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
set_target_properties( ${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/out" )

MESSAGE( STATUS "OpenCV_LIBS: " ${OpenCV_LIBS})
MESSAGE( STATUS "SRC_LIST: " ${SRC_LIST})

#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#	message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

# tell CMake to search first in directories listed in CMAKE_MODULE_PATH
# when you use FIND_PACKAGE() or INCLUDE()
#MESSAGE( STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH} )


