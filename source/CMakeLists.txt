cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ssd-app)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

aux_source_directory(. DIR_SRCS)

add_executable(ssd-app ${DIR_SRCS})
target_link_libraries(ssd-app "${OpenCV_LIBS}")
set_property(TARGET ssd-app PROPERTY CXX_STANDARD 11)
