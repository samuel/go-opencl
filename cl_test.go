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

	devices, err := GetDevices(DeviceTypeAll)
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
		t.Logf("  Name: %+v", d.Name())
		t.Logf("  Vendor: %+v", d.Vendor())
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
	if err := queue.EnqueueWriteBufferFloat32(input, true, 0, data[:]); err != nil {
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
	if err := queue.EnqueueNDRangeKernel(kernel, nil, []int{global}, []int{local}); err != nil {
		t.Fatalf("EnqueueNDRangeKernel failed: %+v", err)
	}

	if err := queue.Finish(); err != nil {
		t.Fatalf("Finish failed: %+v", err)
	}

	results := make([]float32, len(data))
	if err := queue.EnqueueReadBufferFloat32(output, true, 0, results); err != nil {
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
