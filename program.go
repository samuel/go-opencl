package cl

// #include <OpenCL/opencl.h>
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

const errorBufferSize = 2048

type BuildError string

func (e BuildError) Error() string {
	return fmt.Sprintf("cl: build error (%s)", string(e))
}

type Program struct {
	clProgram C.cl_program
	devices   []*Device
}

func releaseProgram(p *Program) {
	if p.clProgram != nil {
		C.clReleaseProgram(p.clProgram)
		p.clProgram = nil
	}
}

func (p *Program) BuildProgram(devices []*Device, options string) error {
	var cOptions *C.char
	if options != "" {
		cOptions = C.CString(options)
		defer C.free(unsafe.Pointer(cOptions))
	}
	var deviceList []C.cl_device_id
	var deviceListPtr *C.cl_device_id
	numDevices := C.cl_uint(0)
	if devices != nil && len(devices) > 0 {
		deviceList = buildDeviceIdList(devices)
		deviceListPtr = &deviceList[0]
	}
	if err := C.clBuildProgram(p.clProgram, numDevices, deviceListPtr, cOptions, nil, nil); err != C.CL_SUCCESS {
		var buffer [errorBufferSize]C.char
		var bLen C.size_t
		C.clGetProgramBuildInfo(p.clProgram, p.devices[0].id, C.CL_PROGRAM_BUILD_LOG, errorBufferSize, unsafe.Pointer(&buffer), &bLen)
		return BuildError(C.GoStringN((*C.char)(unsafe.Pointer(&buffer)), C.int(bLen)))
	}
	return nil
}

func (p *Program) CreateKernel(name string) (*Kernel, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var err C.cl_int
	clKernel := C.clCreateKernel(p.clProgram, cName, &err)
	if err != C.CL_SUCCESS {
		return nil, CLError(err)
	}
	kernel := &Kernel{clKernel: clKernel, name: name}
	runtime.SetFinalizer(kernel, releaseKernel)
	return kernel, nil
}
