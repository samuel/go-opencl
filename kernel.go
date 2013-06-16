package cl

// #include <OpenCL/opencl.h>
import "C"

import (
	"unsafe"
)

type Kernel struct {
	clKernel C.cl_kernel
	name     string
}

func releaseKernel(k *Kernel) {
	if k.clKernel != nil {
		C.clReleaseKernel(k.clKernel)
		k.clKernel = nil
	}
}

func (k *Kernel) SetKernelArg(index, argSize int, arg unsafe.Pointer) error {
	err := C.clSetKernelArg(k.clKernel, C.cl_uint(index), C.size_t(argSize), arg)
	if err != C.CL_SUCCESS {
		return CLError(err)
	}
	return nil
}

func (k *Kernel) SetKernelArgBuffer(index int, buffer *Buffer) error {
	return k.SetKernelArg(index, int(unsafe.Sizeof(buffer.clBuffer)), unsafe.Pointer(&buffer.clBuffer))
}

func (k *Kernel) SetKernelArgUint32(index int, val uint32) error {
	return k.SetKernelArg(index, int(unsafe.Sizeof(val)), unsafe.Pointer(&val))
}

func (k *Kernel) WorkGroupSize(device *Device) (int, error) {
	var size C.size_t
	err := C.clGetKernelWorkGroupInfo(k.clKernel, device.id, C.CL_KERNEL_WORK_GROUP_SIZE, C.size_t(unsafe.Sizeof(size)), unsafe.Pointer(&size), nil)
	if err != C.CL_SUCCESS {
		return 0, CLError(err)
	}
	return int(size), nil
}
