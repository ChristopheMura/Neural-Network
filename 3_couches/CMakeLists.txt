cmake_minimum_required(VERSION 3.20)
project(my_project)

set(CMAKE_CXX_STANDARD 17)

# Trouver Python 3 et NumPy
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NumPy_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NumPy_RESULT
)

# Inclure les headers Python
#include_directories(${Python3_INCLUDE_DIRS} include)
#include_directories(${NumPy_INCLUDE_DIR})

#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/matplotlib-cpp)

# Ajouter l’exécutable
add_executable(run src/main4.cpp src/mylib.cpp)
#add_executable(run src/main2.cpp)

target_include_directories(run PRIVATE 
    ${Python3_INCLUDE_DIRS}
    ${NumPy_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/matplotlib-cpp  # Ajout pour matplotlib-cpp
)

# Lier les librairies Python
target_link_libraries(run ${Python3_LIBRARIES})
