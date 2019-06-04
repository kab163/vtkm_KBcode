#include <vtkm/io/reader/VTKDataSetReader.h>
#include <vtkm/io/writer/VTKDataSetWriter.h>
#include <vtkm/filter/MarchingCubes.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/ColorTable.h>

#include <vtkm/rendering/Camera.h>
#include <vtkm/rendering/View3D.h>
#include <vtkm/rendering/Scene.h>
#include <vtkm/rendering/Color.h>
#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/MapperRayTracer.h>
#include <vtkm/rendering/Mapper.h>
#include <vtkm/rendering/Canvas.h>
//#include <vtkm/rendering/internal/OpenGLHeaders.h> //Required for compile....

vtkm::cont::DataSet OpenDataFromVTKFile()
{
  vtkm::io::reader::VTKDataSetReader reader("proj6B.vtk");  
  return reader.ReadDataSet();
}

void SaveDataAsVTKFile(vtkm::cont::DataSet data) 
{
  vtkm::io::writer::VTKDataSetWriter writer("output.vtk");
  writer.WriteDataSet(data); 
}

int main() {
  //open input file
  vtkm::cont::DataSet input;
  input = OpenDataFromVTKFile();

  vtkm::Range range;
  input.GetPointField("hardyglobal").GetRange(&range);

  vtkm::Float64 isovalue = range.Center();

  //do marching cubes here
  vtkm::filter::MarchingCubes marchingCubes;
  marchingCubes.SetMergeDuplicatePoints(false);
  marchingCubes.SetActiveField("hardyglobal");
  marchingCubes.SetIsoValue(0, isovalue);

  //execute marching cubes
  vtkm::cont::DataSet isosurface = marchingCubes.Execute(input);

  // compute the bounds and extends of the input data
  vtkm::Bounds coordsBounds = input.GetCoordinateSystem().GetBounds();
  vtkm::Vec<vtkm::Float64,3> totalExtent( coordsBounds.X.Length(),
                                          coordsBounds.Y.Length(),
                                          coordsBounds.Z.Length() );
  vtkm::Float64 mag = vtkm::Magnitude(totalExtent);
  vtkm::Normalize(totalExtent);

  // setup a camera and point it to towards the center of the input data
  vtkm::rendering::Camera camera;
  camera.ResetToBounds(coordsBounds);
  camera.SetLookAt(totalExtent*(mag * .5f));
  camera.SetViewUp(vtkm::make_Vec(0.f, 1.f, 0.f));
  camera.SetClippingRange(1.f, 100.f);
  camera.SetFieldOfView(60.f);
  camera.SetPosition(totalExtent*(mag * 2.f));
  vtkm::cont::ColorTable colorTable("inferno");
  
  // Create a mapper, canvas and view that will be used to render the scene
  vtkm::rendering::Scene scene;
  vtkm::rendering::MapperRayTracer mapper;
  vtkm::rendering::CanvasRayTracer canvas(512, 512);
  vtkm::rendering::Color bg(0.2f, 0.2f, 0.2f, 1.0f);
  
  // Render an image of the output isosurface
  scene.AddActor(vtkm::rendering::Actor(isosurface.GetCellSet(),
                                        isosurface.GetCoordinateSystem(),
                                        isosurface.GetField("hardyglobal"),
                                        colorTable));
  vtkm::rendering::View3D view(scene, mapper, canvas, camera, bg);
  view.Initialize();
  view.Paint();
  view.SaveAs("demo_output.png");

  //output the result
  SaveDataAsVTKFile(isosurface);

  return 0;
}
