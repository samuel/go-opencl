package cl

import (
	"math/rand"
	"testing"
)

var kernelSource = `
__kernel void square(
   __global float* input,
   __global float* output,
   const unsigned int count)
{
   int i = get_global_id(0);
   if(i < count)
       output[i] = input[i] * input[i];
}
`

func TestHello(t *testing.T) {
	var data [1024]float32
	for i := 0; i < len(data); i++ {
		data[i] = rand.Float32()
	}

	platforms, err := GetPlatforms()
	if err != nil {
		t.Fatalf("Failed to get platforms: %+v", err)
	}
	for i, p := range platforms {
		t.Logf("Platform %d:", i)
		t.Logf("  Name: %s", p.Name())
		t.Logf("  Vendor: %s", p.Vendor())
		t.Logf("  Profile: %s", p.Profile())
		t.Logf("  Version: %s", p.Version())
		t.Logf("  Extensions: %s", p.Extensions())
	}
	platform := platforms[0]

	devices, err := platform.GetDevices(DeviceTypeAll)
	if err != nil {
		t.Fatalf("Failed to get devices: %+v", err)
	}
	if len(devices) == 0 {
		t.Fatalf("GetDevices returned no devices")
	}
	deviceIndex := -1
	for i, d := range devices {
		if deviceIndex < 0 && d.Type() == DeviceTypeGPU {
			deviceIndex = i
		}
		t.Logf("Device %d:", i)
		t.Logf("  Type: %s", d.Type())
		t.Logf("  Name: %s", d.Name())
		t.Logf("  Vendor: %s", d.Vendor())
		t.Logf("  Built-In Kernels: %s", d.BuiltInKernels())
		t.Logf("  Extensions: %s", d.Extensions())
		t.Logf("  OpenCL C Version: %s", d.OpenCLCVersion())
		t.Logf("  Profile: %s", d.Profile())
		t.Logf("  Version: %s", d.Version())
		t.Logf("  Driver Version: %s", d.DriverVersion())
		t.Logf("  Address Bits: %d", d.AddressBits())
		t.Logf("  Global Memory Cacheline Size: %d", d.GlobalMemCachelineSize())
		t.Logf("  Max Clock Frequency: %d", d.MaxClockFrequency())
		t.Logf("  Max Compute Units: %d", d.MaxComputeUnits())
		t.Logf("  Max Constant Args: %d", d.MaxConstantArgs())
		t.Logf("  Max Parameter Size: %d", d.MaxParameterSize())
		t.Logf("  Max Read-Image Args: %d", d.MaxReadImageArgs())
		t.Logf("  Max Samplers: %d", d.MaxSamplers())
		t.Logf("  Max Work Group Size: %d", d.MaxWorkGroupSize())
		t.Logf("  Max Work Item Dimensions: %d", d.MaxWorkItemDimensions())
		t.Logf("  Max Write-Image Args: %d", d.MaxWriteImageArgs())
		t.Logf("  Memory Base Address Alignment: %d", d.MemBaseAddrAlign())
		t.Logf("  Image2D Max Dimensions: %d x %d", d.Image2DMaxWidth(), d.Image2DMaxHeight())
		t.Logf("  Available: %+v", d.Available())
		t.Logf("  Compiler Available: %+v", d.CompilerAvailable())
		t.Logf("  Little Endian: %+v", d.EndianLittle())
		t.Logf("  Error Correction Supported: %+v", d.ErrorCorrectionSupport())
		t.Logf("  Host Unified Memory: %+v", d.HostUnifiedMemory())
		t.Logf("  Image Support: %+v", d.ImageSupport())
		t.Logf("  Double FP Config: %s", d.DoubleFPConfig())
		t.Logf("  Half FP Config: %s", d.HalfFPConfig())
	}
	if deviceIndex < 0 {
		deviceIndex = 0
	}
	device := devices[deviceIndex]
	t.Logf("Using device %d", deviceIndex)
	context, err := CreateContext([]*Device{device})
	if err != nil {
		t.Fatalf("CreateContext failed: %+v", err)
	}
	// imageFormats, err := context.GetSupportedImageFormats(0, MemObjectTypeImage2D)
	// if err != nil {
	// 	t.Fatalf("GetSupportedImageFormats failed: %+v", err)
	// }
	// t.Logf("Supported image formats: %+v", imageFormats)
	queue, err := context.CreateCommandQueue(device, 0)
	if err != nil {
		t.Fatalf("CreateCommandQueue failed: %+v", err)
	}
	program, err := context.CreateProgramWithSource([]string{kernelSource})
	if err != nil {
		t.Fatalf("CreateProgramWithSource failed: %+v", err)
	}
	if err := program.BuildProgram(nil, ""); err != nil {
		t.Fatalf("BuildProgram failed: %+v", err)
	}
	kernel, err := program.CreateKernel("square")
	if err != nil {
		t.Fatalf("CreateKernel failed: %+v", err)
	}
	input, err := context.CreateBuffer(MemReadOnly, 4*len(data), nil)
	if err != nil {
		t.Fatalf("CreateBuffer failed for input: %+v", err)
	}
	output, err := context.CreateBuffer(MemReadOnly, 4*len(data), nil)
	if err != nil {
		t.Fatalf("CreateBuffer failed for output: %+v", err)
	}
	if _, err := queue.EnqueueWriteBufferFloat32(input, true, 0, data[:], nil); err != nil {
		t.Fatalf("EnqueueWriteBufferFloat32 failed: %+v", err)
	}
	if err := kernel.SetKernelArgBuffer(0, input); err != nil {
		t.Fatalf("SetKernelArgBuffer failed for input: %+v", err)
	}
	if err := kernel.SetKernelArgBuffer(1, output); err != nil {
		t.Fatalf("SetKernelArgBuffer failed for output: %+v", err)
	}
	if err := kernel.SetKernelArgUint32(2, uint32(len(data))); err != nil {
		t.Fatalf("SetKernelArgBuffer failed for count: %+v", err)
	}

	local, err := kernel.WorkGroupSize(device)
	if err != nil {
		t.Fatalf("WorkGroupSize failed: %+v", err)
	}
	t.Logf("Work group size: %d", local)

	global := len(data)
	d := len(data) % local
	if d != 0 {
		global += local - d
	}
	if _, err := queue.EnqueueNDRangeKernel(kernel, nil, []int{global}, []int{local}, nil); err != nil {
		t.Fatalf("EnqueueNDRangeKernel failed: %+v", err)
	}

	if err := queue.Finish(); err != nil {
		t.Fatalf("Finish failed: %+v", err)
	}

	results := make([]float32, len(data))
	if _, err := queue.EnqueueReadBufferFloat32(output, true, 0, results, nil); err != nil {
		t.Fatalf("EnqueueReadBufferFloat32 failed: %+v", err)
	}

	correct := 0
	for i, v := range data {
		if results[i] == v*v {
			correct++
		}
	}

	if correct != len(data) {
		t.Fatalf("%d/%d correct values", correct, len(data))
	}
}
