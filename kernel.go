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

func (k *Kernel) Release() {
	releaseKernel(k)
}

func (k *Kernel) SetKernelArg(index, argSize int, arg unsafe.Pointer) error {
	return toError(C.clSetKernelArg(k.clKernel, C.cl_uint(index), C.size_t(argSize), arg))
}

func (k *Kernel) SetKernelArgBuffer(index int, buffer *MemObject) error {
	return k.SetKernelArg(index, int(unsafe.Sizeof(buffer.clMem)), unsafe.Pointer(&buffer.clMem))
}

func (k *Kernel) SetKernelArgInt32(index int, val int32) error {
	return k.SetKernelArg(index, int(unsafe.Sizeof(val)), unsafe.Pointer(&val))
}

func (k *Kernel) SetKernelArgUint32(index int, val uint32) error {
	return k.SetKernelArg(index, int(unsafe.Sizeof(val)), unsafe.Pointer(&val))
}

func (k *Kernel) PreferredWorkGroupSizeMultiple(device *Device) (int, error) {
	var size C.size_t
	err := C.clGetKernelWorkGroupInfo(k.clKernel, device.nullableId(), C.CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, C.size_t(unsafe.Sizeof(size)), unsafe.Pointer(&size), nil)
	return int(size), toError(err)
}

func (k *Kernel) WorkGroupSize(device *Device) (int, error) {
	var size C.size_t
	err := C.clGetKernelWorkGroupInfo(k.clKernel, device.nullableId(), C.CL_KERNEL_WORK_GROUP_SIZE, C.size_t(unsafe.Sizeof(size)), unsafe.Pointer(&size), nil)
	return int(size), toError(err)
}

func (k *Kernel) NumArgs() (int, error) {
	var num C.cl_uint
	err := C.clGetKernelInfo(k.clKernel, C.CL_KERNEL_NUM_ARGS, C.size_t(unsafe.Sizeof(num)), unsafe.Pointer(&num), nil)
	return int(num), toError(err)
}

// func (k *Kernel) ArgName(index int) (string, error) {
// 	var strC [1024]byte
// 	var strN C.size_t
// 	if err := C.getKernelArgInfo(k.clKernel, CL_KERNEL_ARG_NAME, 1024, unsafe.Pointer(&strC[0]), &strN); err != C.CL_SUCCESS {
// 		return "", toError(err)
// 	}
// 	return string(strC[:strN]), nil
// }
