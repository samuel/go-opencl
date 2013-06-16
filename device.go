package cl

// #include <OpenCL/opencl.h>
import "C"

import (
	"strings"
	"unsafe"
)

type DeviceType int

const maxDeviceCount = 64

const (
	DeviceTypeCPU         DeviceType = C.CL_DEVICE_TYPE_CPU
	DeviceTypeGPU         DeviceType = C.CL_DEVICE_TYPE_GPU
	DeviceTypeAccelerator DeviceType = C.CL_DEVICE_TYPE_ACCELERATOR
	DeviceTypeDefault     DeviceType = C.CL_DEVICE_TYPE_DEFAULT
	DeviceTypeAll         DeviceType = C.CL_DEVICE_TYPE_ALL
)

func (dt DeviceType) String() string {
	var parts []string
	if dt&DeviceTypeCPU != 0 {
		parts = append(parts, "CPU")
	}
	if dt&DeviceTypeGPU != 0 {
		parts = append(parts, "GPU")
	}
	if dt&DeviceTypeAccelerator != 0 {
		parts = append(parts, "Accelerator")
	}
	if dt&DeviceTypeDefault != 0 {
		parts = append(parts, "Default")
	}
	if parts == nil {
		parts = append(parts, "None")
	}
	return strings.Join(parts, "|")
}

type Device struct {
	id C.cl_device_id
}

func buildDeviceIdList(devices []*Device) []C.cl_device_id {
	deviceIds := make([]C.cl_device_id, len(devices))
	for i, d := range devices {
		deviceIds[i] = d.id
	}
	return deviceIds
}

func GetDevices(deviceType DeviceType) ([]*Device, error) {
	var deviceIds [maxDeviceCount]C.cl_device_id
	var numDevices C.cl_uint
	if err := C.clGetDeviceIDs(nil, C.cl_device_type(deviceType), C.cl_uint(maxDeviceCount), &deviceIds[0], &numDevices); err != C.CL_SUCCESS {
		return nil, CLError(int(err))
	}
	if numDevices > maxDeviceCount {
		numDevices = maxDeviceCount
	}
	devices := make([]*Device, numDevices)
	for i := 0; i < int(numDevices); i++ {
		devices[i] = &Device{id: deviceIds[i]}
	}
	return devices, nil
}

func (d *Device) Name() string {
	var deviceNameC [256]C.char
	var deviceNameN C.size_t
	if err := C.clGetDeviceInfo(d.id, C.CL_DEVICE_NAME, 256, unsafe.Pointer(&deviceNameC), &deviceNameN); err != C.CL_SUCCESS {
		// Should never fail
		panic("Failed to get device name")
	}
	return C.GoStringN((*C.char)(unsafe.Pointer(&deviceNameC)), C.int(deviceNameN))
}

func (d *Device) Vendor() string {
	var vendorC [256]C.char
	var vendorN C.size_t
	if err := C.clGetDeviceInfo(d.id, C.CL_DEVICE_VENDOR, 256, unsafe.Pointer(&vendorC), &vendorN); err != C.CL_SUCCESS {
		// Should never fail
		panic("Failed to get device vendor")
	}
	return C.GoStringN((*C.char)(unsafe.Pointer(&vendorC)), C.int(vendorN))
}

func (d *Device) Type() DeviceType {
	var deviceType C.cl_device_type
	if err := C.clGetDeviceInfo(d.id, C.CL_DEVICE_TYPE, C.size_t(unsafe.Sizeof(deviceType)), unsafe.Pointer(&deviceType), nil); err != C.CL_SUCCESS {
		// Should never fail
		panic("Failed to get device type")
	}
	return DeviceType(deviceType)
}
