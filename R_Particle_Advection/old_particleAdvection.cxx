#include <iostream>
#include <fstream>

//----------------------------------------------------



#include <vtkm/worklet/particleadvection/ParticleAdvectionWorklets.h>
#include <vtkm/worklet/particleadvection/Particles.h>

#include <vtkm/worklet/ParticleAdvection.h>
#include <vtkm/worklet/particleadvection/GridEvaluators.h>
#include <vtkm/worklet/particleadvection/Integrators.h>
#include <vtkm/worklet/particleadvection/ParticleAdvectionWorklets.h>
#include <vtkm/worklet/particleadvection/Particles.h>
#include <vtkm/worklet/StreamLineUniformGrid.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/Types.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/cont/DeviceAdapter.h>
#include <vtkm/cont/RuntimeDeviceTracker.h>

#include <tbb/task_scheduler_init.h>
//#include <omp.h>

#include <time.h>
#include <sys/time.h>
#include "particles_array.h"

#define TIMEINFO timeval
#define THRESHOLD 1
//----------------------------------------------------
using namespace std;

//setting up the device adapter tags.
using cpu = vtkm::cont::DeviceAdapterTagTBB;
using gpu = vtkm::cont::DeviceAdapterTagCuda;

void DeviceOracle(const int num) { 	
  if(num < THRESHOLD) {
    vtkm::cont::GetRuntimeDeviceTracker().ForceDevice(gpu());//tracker.ForceDevice(gpu());
  } else {
    vtkm::cont::GetRuntimeDeviceTracker().ForceDevice(cpu());
  }
}

//----------------------------------------------------
double
DiffTime(const struct TIMEINFO &startTime,
                         const struct TIMEINFO &endTime)
{
    double seconds = double(endTime.tv_sec - startTime.tv_sec) +
                     double(endTime.tv_usec - startTime.tv_usec) / 1000000.;

    return seconds;
}
//----------------------------------------------------
int main(int argc, char **argv)
{
  const int DimNum = 3; //3 demensional algorithm - value related to dim_xyz
  //const int OUT_NUM = 1024;

  if(argc != 6) std::cerr<< "usage: <exe> fieldnum nP num_steps step_size dim_xyz";
  const int fieldnum = std::atoi(argv[1]); //value used when creating the velocity field
  const int nP = std::atoi(argv[2]); //number of particles
  const int nSteps = std::atoi(argv[3]); //number of steps the particles will advect
  const double stepSize = std::atoi(argv[4]); //how big each step is
  const int dim_xyz = std::atoi(argv[5]); //dimensions (related to fieldnum)
  
  std::vector<vtkm::Vec<vtkm::Float32, DimNum> >vtkmVec2;
  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, DimNum> > field;

  vtkm::cont::DataSetBuilderUniform dataSetBuilder;
  vtkm::Vec<vtkm::Float32, DimNum> origin(0,0,0);
  vtkm::Vec<vtkm::Float32, DimNum> spacing(1,1,1);
  vtkm::Id3 dims(dim_xyz, dim_xyz, dim_xyz);
  vtkm::cont::DataSet ds;
  ds = dataSetBuilder.Create(dims, origin, spacing);
  std::cout<<"dataSetBuilder done \n";

  #if 1
  int numThreads = tbb::task_scheduler_init::automatic;
  numThreads = 12;
  tbb::task_scheduler_init init(numThreads);
  #endif

  //omp_set_num_threads(12);

  for (int i = 0; i < fieldnum; i++)
  {   
    vtkm::Float32 x = 0;
    vtkm::Float32 y = 0;
    vtkm::Float32 z = 1;
    vtkmVec2.push_back(vtkm::Vec<vtkm::Float32, DimNum>(x,y,z));
  } 
  field = vtkm::cont::make_ArrayHandle(vtkmVec2);

  ds.AddField(vtkm::cont::Field("vecData", vtkm::cont::Field::Association::POINTS, field));
  typedef vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, DimNum>> FieldHandle;
  using FieldType = vtkm::Float32;
  using RGEvalType = vtkm::worklet::particleadvection::GridEvaluator<FieldHandle>;
  using RK4RGType = vtkm::worklet::particleadvection::RK4Integrator<RGEvalType>;

  RGEvalType eval(ds.GetCoordinateSystem(), ds.GetCellSet(0), field);
  RK4RGType rk4(eval, stepSize);

  std::vector<vtkm::Vec<vtkm::Float32, DimNum>> seeds;
  std::vector<vtkm::Id> particleSteps;
  vtkm::Id pSteps;

  for(int i=0; i <nP; i++)
  {
    vtkm::Float32 x = particles[i][0];
    vtkm::Float32 y = particles[i][1];
    vtkm::Float32 z = particles[i][2];
    seeds.push_back(vtkm::Vec<vtkm::Float32, DimNum>(x,y,z));
    pSteps = 0;
    particleSteps.push_back(pSteps);
  }	 
  
  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, DimNum>> seedArray;
  seedArray = vtkm::cont::make_ArrayHandle(seeds);
  
  vtkm::cont::ArrayHandle< vtkm::Id > particleStepsArray;
  particleStepsArray = vtkm::cont::make_ArrayHandle(particleSteps);
  
  struct TIMEINFO startTime2, endTime2; 
  vtkm::worklet::ParticleAdvectionResult res;
  vtkm::worklet::ParticleAdvection particleAdvection;
 
  DeviceOracle(nP);
  //vtkm::worklet::DispatcherMapField<vtkm::worklet::ParticleAdvection> paDispatcher(particleAdvection);
  //paDispatcher.SetDevice(gpu());
  
  gettimeofday(&startTime2, 0);
  res = particleAdvection.Run(rk4, seedArray, particleStepsArray, nSteps);
  gettimeofday(&endTime2, 0);

  double t2 = DiffTime(startTime2, endTime2);
  cout << "The time of Actual particle vtkm call is " << t2 << endl;
  //cout << "Result of PA is " << res << endl;

  return 0;   
}	

