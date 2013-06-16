package cl

// #include <OpenCL/opencl.h>
import "C"

import (
	"unsafe"
)

const maxPlatforms = 32

type Platform struct {
	id C.cl_platform_id
}

func GetPlatforms() ([]*Platform, error) {
	var platformIds [maxPlatforms]C.cl_platform_id
	var nPlatforms C.cl_uint
	if err := C.clGetPlatformIDs(C.cl_uint(maxPlatforms), &platformIds[0], &nPlatforms); err != C.CL_SUCCESS {
		return nil, CLError(err)
	}
	platforms := make([]*Platform, nPlatforms)
	for i := 0; i < int(nPlatforms); i++ {
		platforms[i] = &Platform{id: platformIds[i]}
	}
	return platforms, nil
}

func (p *Platform) GetDevices(deviceType DeviceType) ([]*Device, error) {
	return GetDevices(p, deviceType)
}

func (p *Platform) getInfoString(param C.cl_platform_info) (string, error) {
	var strC [1024]C.char
	var strN C.size_t
	if err := C.clGetPlatformInfo(p.id, param, 1024, unsafe.Pointer(&strC), &strN); err != C.CL_SUCCESS {
		return "", CLError(err)
	}
	return C.GoStringN((*C.char)(unsafe.Pointer(&strC)), C.int(strN)), nil
}

func (p *Platform) Name() string {
	if str, err := p.getInfoString(C.CL_PLATFORM_NAME); err != nil {
		panic("Platform.Name() should never fail")
	} else {
		return str
	}
}

func (p *Platform) Vendor() string {
	if str, err := p.getInfoString(C.CL_PLATFORM_VENDOR); err != nil {
		panic("Platform.Vendor() should never fail")
	} else {
		return str
	}
}

func (p *Platform) Profile() string {
	if str, err := p.getInfoString(C.CL_PLATFORM_PROFILE); err != nil {
		panic("Platform.Profile() should never fail")
	} else {
		return str
	}
}

func (p *Platform) Version() string {
	if str, err := p.getInfoString(C.CL_PLATFORM_VERSION); err != nil {
		panic("Platform.Version() should never fail")
	} else {
		return str
	}
}

func (p *Platform) Extensions() string {
	if str, err := p.getInfoString(C.CL_PLATFORM_EXTENSIONS); err != nil {
		panic("Platform.Extensions() should never fail")
	} else {
		return str
	}
}
