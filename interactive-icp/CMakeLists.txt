cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pcl-interactive_icp)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
add_executable (tutorial_icp tutorial_icp.cpp)
target_link_libraries (tutorial_icp ${PCL_LIBRARIES} )
