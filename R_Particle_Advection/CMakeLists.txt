cmake_minimum_required(VERSION 3.10)
project(particleAdvect)


SET(CMAKE_C_COMPILER gcc)
SET(CMAKE_CXX_COMPILER g++)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -std=c++11 -O3 -fopenmp")


include_directories(.)

find_package(VTKm REQUIRED  OPTIONAL_COMPONENTS Rendering OPTIONAL_COMPONENTS TBB CUDA)

if ( NOT VTKm_ENABLE_TBB )
       message ( SEND_ERROR " VTK -m must be built with TBB .")
endif ()



add_executable(particles testInteg.cxx)



target_include_directories(particles PRIVATE ${VTKm_INCLUDE_DIRS})
target_link_libraries(particles PRIVATE vtkm_cont vtkm_filter ${VTKm_LIBRARIES})
target_compile_options(particles PRIVATE ${VTKm_COMPILE_OPTIONS})

if(TARGET vtkm::cuda)
	set_source_files_properties(testInteg.cxx PROPERTIES LANGUAGE "CUDA")
endif()
