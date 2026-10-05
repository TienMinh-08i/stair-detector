cmake_minimum_required(VERSION 3.16)
project(orbbec_opencv_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED)

find_package(PCL REQUIRED COMPONENTS
    common
    features
    io
    filters
    segmentation
)

set(ORBBEC_ROOT "$ENV{HOME}/Downloads/OrbbecSDK_v1.10.37"
    CACHE PATH "Path to OrbbecSDK root")

find_path(ORBBEC_INCLUDE_DIR
    NAMES libobsensor/ObSensor.hpp
    PATHS
        "${ORBBEC_ROOT}/include"
        "${ORBBEC_ROOT}/SDK/include"
        "${ORBBEC_ROOT}"
    PATH_SUFFIXES include SDK/include
)

find_library(ORBBEC_LIBRARY
    NAMES OrbbecSDK
    PATHS
        "${ORBBEC_ROOT}/lib"
        "${ORBBEC_ROOT}/SDK/lib"
        "${ORBBEC_ROOT}/Example/bin"
    NO_DEFAULT_PATH
)

if(NOT ORBBEC_LIBRARY)
    find_file(ORBBEC_LIBRARY
        NAMES
            libOrbbecSDK.so.1.10.37
            libOrbbecSDK.so
        PATHS
            "${ORBBEC_ROOT}/lib"
            "${ORBBEC_ROOT}/SDK/lib"
            "${ORBBEC_ROOT}/Example/bin"
        NO_DEFAULT_PATH
    )
endif()

if(NOT ORBBEC_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find OrbbecSDK headers")
endif()

if(NOT ORBBEC_LIBRARY)
    message(FATAL_ERROR "Could not find OrbbecSDK library")
endif()

include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/capture_view.cpp")
    add_executable(capture_view capture_view.cpp)
    target_include_directories(capture_view PRIVATE
        ${ORBBEC_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(capture_view PRIVATE
        ${ORBBEC_LIBRARY}
        ${OpenCV_LIBS}
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/detect_planes_opencv.cpp")
    add_executable(detect_planes_opencv detect_planes_opencv.cpp)
    target_include_directories(detect_planes_opencv PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
    )
    target_link_libraries(detect_planes_opencv PRIVATE
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
    )
endif()

add_executable(realtime_plane_detector
    realtime_plane_detector.cpp
)

target_compile_definitions(realtime_plane_detector PRIVATE
    EIGEN_NO_DEBUG
)

target_compile_options(realtime_plane_detector PRIVATE
    $<$<CONFIG:Release>:-O3>
)

target_include_directories(realtime_plane_detector PRIVATE
    ${ORBBEC_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(realtime_plane_detector PRIVATE
    ${ORBBEC_LIBRARY}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)

add_executable(realtime_plane_detector_locked_floor
    realtime_plane_detector_locked_floor.cpp
)

target_compile_definitions(realtime_plane_detector_locked_floor PRIVATE
    EIGEN_NO_DEBUG
)

target_compile_options(realtime_plane_detector_locked_floor PRIVATE
    $<$<CONFIG:Release>:-O3>
)

target_include_directories(realtime_plane_detector_locked_floor PRIVATE
    ${ORBBEC_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(realtime_plane_detector_locked_floor PRIVATE
    ${ORBBEC_LIBRARY}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)

add_executable(realtime_stair_detector
    realtime_stair_detector.cpp
)

target_compile_definitions(realtime_stair_detector PRIVATE
    EIGEN_NO_DEBUG
)

target_compile_options(realtime_stair_detector PRIVATE
    $<$<CONFIG:Release>:-O3>
)

target_include_directories(realtime_stair_detector PRIVATE
    ${ORBBEC_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(realtime_stair_detector PRIVATE
    ${ORBBEC_LIBRARY}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)
