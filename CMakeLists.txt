cmake_minimum_required(VERSION 3.14.5 FATAL_ERROR)
project(sgDetector)

find_package(OpenCV REQUIRED)
find_package(RapidJSON REQUIRED)


include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${RapidJSON_INCLUDE_DIRS}
        /home/ierturk/Work/Libs/libonnxrt/include
        Source/Inc
        Source
)

set(SRC
        # Source/Src/sgDetector.cpp
        # Source/Src/OrtNet.cpp
        # Source/Src/ProcessChannel.cpp
        # Source/Src/DataLoader.cpp
        Source/main.cpp
        Source/OrtNet.cpp
        )
set(INC
        # Source/Inc/OrtNet.h
        # Source/Inc/ProcessChannel.h
        # Source/Inc/QueueFPS.h
        # Source/Inc/DataLoader.h
        Source/OrtNet.h
        )

link_directories(
        /home/ierturk/Work/Libs/libonnxrt/lib
)
add_executable(${PROJECT_NAME} ${SRC} ${INC})

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        onnxruntime
        pthread
        )