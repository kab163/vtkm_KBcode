cmake_minimum_required(VERSION 3.8...3.14 FATAL_ERROR)                          
project(SwitchDevice CXX)                                                  
                                                                                
#Find the VTK-m package                                                         
find_package(VTKm REQUIRED QUIET)                                               
                                                                                
add_executable(SwitchDevice SwitchDevice.cxx)                        
target_link_libraries(SwitchDevice PRIVATE vtkm_cont vtkm_worklet) 
                                                                                
if(TARGET vtkm::tbb)                                                            
  target_compile_definitions(SwitchDevice PRIVATE BUILDING_TBB_VERSION)   
endif()                                                                         
                                                                                
if(TARGET vtkm::cuda)                                                           
  set_source_files_properties(SwitchDevice.cxx PROPERTIES LANGUAGE "CUDA") 
endif()  
