#include <vector>

#include <vtkm/cont/DeviceAdapter.h>

#include <vtkm/worklet/DispatcherMapField.h>

namespace
{

class Square : public vtkm::worklet::WorkletMapField
{
public:
  Square() = default;

  using ControlSignature = void(FieldIn, FieldOut);
  using ExecutionSignature = void(_1, _2);

  template <typename ScalarType>
  VTKM_EXEC
  void operator()(const ScalarType input,
                  ScalarType& output) const
  {
    output = input * input;
  }
};

}


int main(int argc, char** argv)
{
  std::vector<vtkm::FloatDefault> numbers;
  for(size_t index = 0; index < 20; index++)
  {
    numbers.push_back(index);
  }

  vtkm::cont::ArrayHandle<vtkm::FloatDefault> input = vtkm::cont::make_ArrayHandle(numbers);
  vtkm::cont::ArrayHandle<vtkm::FloatDefault> cpuOutput;
  vtkm::cont::ArrayHandle<vtkm::FloatDefault> gpuOutput;

  using cpu = vtkm::cont::DeviceAdapterTagTBB;
  using gpu = vtkm::cont::DeviceAdapterTagCuda;

  vtkm::worklet::DispatcherMapField<Square> dispatcher;

  dispatcher.SetDevice(cpu());
  dispatcher.Invoke(input, cpuOutput);

  dispatcher.SetDevice(gpu());
  dispatcher.Invoke(input, gpuOutput);

  auto inPortal = input.GetPortalConstControl();
  auto cpuPortal = cpuOutput.GetPortalConstControl();
  auto gpuPortal = gpuOutput.GetPortalConstControl();

  std::cout << "CPU output" << std::endl;
  for(size_t index = 0; index < 20; index++)
  {
    std::cout << inPortal.Get(index) << " : " << cpuPortal.Get(index) << std::endl;
  }

  std::cout << "GPU output" << std::endl;
  for(size_t index = 0; index < 20; index++)
  {
    std::cout << inPortal.Get(index) << " : " << gpuPortal.Get(index) << std::endl;
  }

  return 0;
}
