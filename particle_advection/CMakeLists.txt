##============================================================================
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##============================================================================
cmake_minimum_required(VERSION 3.8...3.14 FATAL_ERROR)
project(ParticleAdvection CXX)

#Find the VTK-m package
find_package(VTKm REQUIRED QUIET)

add_executable(ParticleAdvection ParticleAdvection.cxx)
target_link_libraries(ParticleAdvection PRIVATE vtkm_filter)

if(TARGET vtkm::tbb)
  target_compile_definitions(ParticleAdvection PRIVATE BUILDING_TBB_VERSION)
endif()

if(TARGET vtkm::cuda)
  set_source_files_properties(ParticleAdvection.cxx PROPERTIES LANGUAGE "CUDA")
endif()
