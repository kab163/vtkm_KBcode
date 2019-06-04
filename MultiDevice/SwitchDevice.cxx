/*
 * This is a simple vtkm program that utilizes both the GPU and the CPU during execution.
 * It starts with a certain number of input values and squares those numbers for output.
 * Complexity is added with 'rounds' where in each round, either the CPU or the GPU is
 * executing the workload. Next, the number of input values that need to be squared is
 * also varied for complexity. Here, there is a certain threshold so that values below
 * the threshold are computed by the CPU and values above it are executed by the GPU.
 *
 * This program serves as a proof of concept that we can change desired hardware
 * during execution based upon a certain threshold that changes.
 */

#include <vector>
#include <vtkm/cont/DeviceAdapter.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/Math.h>

#define NUM 40
#define ROUNDS 5
#define THRESHOLD 20

/*
 * Simple function to decide if the code should be run on the GPU or not.
 * This is a base version for my Oracle. 
 *
 * Input: GPU ArrayHandle (to access data seen by GPU) - (plus, potentially 
 * other arguments for determining if code should be run on GPU or not.)
 *
 * Return value is boolean - true if it should run on GPU, false otherwise.
 */

bool deviceOracle(vtkm::cont::ArrayHandle<vtkm::FloatDefault> gpuInput, size_t rounds) {
  bool result = false;

  if(gpuInput.GetNumberOfValues() >= (THRESHOLD * rounds)) result = true;

  return result;
}

/*
 * Creating a vtkm worklet that will take an input value and square it.
 * We use the WorkletMapField worklet because we are doing a simple operation 
 * to each element of our ArrayHandle.
 */
namespace
{

class Square : public vtkm::worklet::WorkletMapField
{
public:
  Square() = default; //nothing fancy needs to be done in constructor

  using ControlSignature = void(FieldIn, FieldOut);
  using ExecutionSignature = void(_1, _2);

  template <typename ScalarType> VTKM_EXEC
  void operator()(const ScalarType input, ScalarType& output) const
  {
    output = input * input; //main worklet operation
  }
};

}

int main(int argc, char** argv)
{
  std::vector<vtkm::FloatDefault> numbers; //holds the original input
  std::vector<vtkm::FloatDefault> gEditing; //holds the modified GPU input for each round
  std::vector<vtkm::FloatDefault> cEditing; //holds the modified CPU input for each round
  
  //generating input
  for(size_t index = 0; index < NUM; index++)
  {
    numbers.push_back(index);
  }

  //ArrayHandle that hold GPU and CPU inputs
  vtkm::cont::ArrayHandle<vtkm::FloatDefault> Input = vtkm::cont::make_ArrayHandle(numbers);
  
  //ArrayHandles that hold GPU and CPU outputs
  vtkm::cont::ArrayHandle<vtkm::FloatDefault> cpuOutput;
  vtkm::cont::ArrayHandle<vtkm::FloatDefault> gpuOutput;

  //setting up the device adapter tags.
  using cpu = vtkm::cont::DeviceAdapterTagTBB;
  using gpu = vtkm::cont::DeviceAdapterTagCuda;

  //setting up a vtkm dispatcher specified by map operations on 'Square's 
  vtkm::worklet::DispatcherMapField<Square> dispatcher;

  //setting up the type of portal we want for the ArrayHandles
  using PortalType = typename vtkm::cont::ArrayHandle<vtkm::FloatDefault>::PortalControl;

  //ArrayPortals for CPU and GPU outputs (note this is read/write)
  PortalType gpuPortal = gpuOutput.GetPortalControl();
  PortalType cpuPortal = cpuOutput.GetPortalControl();

  //looping over the specified number of rounds...
  for(size_t rounds = 0; rounds < ROUNDS; rounds++) {

    if(deviceOracle(Input, rounds)) {
      dispatcher.SetDevice(gpu()); //this specifies what device we want to run on
      dispatcher.Invoke(Input, gpuOutput); //calls the code to actually run on the device

      gpuPortal = gpuOutput.GetPortalControl(); //portal for output, now we can access this
      Input.ReleaseResources(); //get rid of old input, get ready for modified input

      //testing if output is above threshold, if it is, it becomes input for next round
      for(size_t gIter = 0; gIter < gpuPortal.GetNumberOfValues(); gIter++)
      {
        if(gpuPortal.Get(gIter) > THRESHOLD) gEditing.push_back(gpuPortal.Get(gIter));
      } 

      Input = vtkm::cont::make_ArrayHandle(gEditing); //setting modified input as our input

      std::cout << "GPU output for Round # : " << rounds << std::endl;
      for(size_t index = 0; index < gpuPortal.GetNumberOfValues(); index++)
      {
        std::cout << index << " : " << gpuPortal.Get(index) << std::endl;
      }
      gEditing.clear();
    }
    else {
      dispatcher.SetDevice(cpu()); //specifies the device we want to run on
      dispatcher.Invoke(Input, cpuOutput); //calls the code to actually run on device
    
      cpuPortal = cpuOutput.GetPortalControl(); //portal for output, now we can access the results
      Input.ReleaseResources(); //getting rid of old input so we can make room for the modified

      //testing if output is below threshold, if it is, it becomes input for next round
      for(size_t iter = 0; iter < cpuPortal.GetNumberOfValues(); iter++)
      {
        if(vtkm::Log(cpuPortal.Get(iter)) < THRESHOLD) cEditing.push_back(numbers.at(iter));
      }

      Input = vtkm::cont::make_ArrayHandle(cEditing); //setting modified input as our input

      std::cout << "CPU output for Round # : " << rounds << std::endl;
      for(size_t index = 0; index < cpuPortal.GetNumberOfValues(); index++)
      {
        std::cout << index << " : " << cpuPortal.Get(index) << std::endl;
      }
      cEditing.clear();
    }
  }

  return 0;
}
