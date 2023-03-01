cmake_minimum_required(VERSION 3.0)
project( mono-vo )

set(CMAKE_BUILD_TYPE Release)

SET("OpenCV_DIR" "/home/ernie/opencv-4.x/build")
find_package( OpenCV REQUIRED )

message(STATUS "OpenCV version: $ {OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")

file(GLOB viso
    "src/*.cpp"
)
add_executable( vo ${viso} )
target_link_libraries( vo ${OpenCV_LIBS})
