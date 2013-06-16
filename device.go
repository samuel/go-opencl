package cl

// #include <OpenCL/opencl.h>
import "C"

import (
	"strings"
	"unsafe"
)

const maxDeviceCount = 64

type DeviceType int

const (
	DeviceTypeCPU         DeviceType = C.CL_DEVICE_TYPE_CPU
	DeviceTypeGPU         DeviceType = C.CL_DEVICE_TYPE_GPU
	DeviceTypeAccelerator DeviceType = C.CL_DEVICE_TYPE_ACCELERATOR
	DeviceTypeDefault     DeviceType = C.CL_DEVICE_TYPE_DEFAULT
	DeviceTypeAll         DeviceType = C.CL_DEVICE_TYPE_ALL
)

type FPConfig int

const (
	FPConfigDenorm                     FPConfig = C.CL_FP_DENORM           // denorms are supported
	FPConfigInfNaN                     FPConfig = C.CL_FP_INF_NAN          // INF and NaNs are supported
	FPConfigRoundToNearest             FPConfig = C.CL_FP_ROUND_TO_NEAREST // round to nearest even rounding mode supported
	FPConfigRoundToZero                FPConfig = C.CL_FP_ROUND_TO_ZERO    // round to zero rounding mode supported
	FPConfigRoundToInf                 FPConfig = C.CL_FP_ROUND_TO_INF     // round to positive and negative infinity rounding modes supported
	FPConfigFMA                        FPConfig = C.CL_FP_FMA              // IEEE754-2008 fused multiply-add is supported
	FPConfigSoftFloat                  FPConfig = C.CL_FP_SOFT_FLOAT       // Basic floating-point operations (such as addition, subtraction, multiplication) are implemented in software
	FPConfigCorrectlyRoundedDivideSqrt FPConfig = C.CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT
)

var fpConfigNameMap = map[FPConfig]string{
	FPConfigDenorm:                     "Denorm",
	FPConfigInfNaN:                     "InfNaN",
	FPConfigRoundToNearest:             "RoundToNearest",
	FPConfigRoundToZero:                "RoundToZero",
	FPConfigRoundToInf:                 "RoundToInf",
	FPConfigFMA:                        "FMA",
	FPConfigSoftFloat:                  "SoftFloat",
	FPConfigCorrectlyRoundedDivideSqrt: "CorrectlyRoundedDivideSqrt",
}

func (c FPConfig) String() string {
	var parts []string
	for bit, name := range fpConfigNameMap {
		if c&bit != 0 {
			parts = append(parts, name)
		}
	}
	if parts == nil {
		return ""
	}
	return strings.Join(parts, "|")
}

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

func GetDevices(platform *Platform, deviceType DeviceType) ([]*Device, error) {
	var deviceIds [maxDeviceCount]C.cl_device_id
	var numDevices C.cl_uint
	var platformId C.cl_platform_id
	if platform != nil {
		platformId = platform.id
	}
	if err := C.clGetDeviceIDs(platformId, C.cl_device_type(deviceType), C.cl_uint(maxDeviceCount), &deviceIds[0], &numDevices); err != C.CL_SUCCESS {
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

func (d *Device) getInfoString(param C.cl_device_info, panicOnError bool) (string, error) {
	var strC [1024]C.char
	var strN C.size_t
	if err := C.clGetDeviceInfo(d.id, param, 1024, unsafe.Pointer(&strC), &strN); err != C.CL_SUCCESS {
		if panicOnError {
			panic("Should never fail")
		}
		return "", CLError(err)
	}
	return C.GoStringN((*C.char)(unsafe.Pointer(&strC)), C.int(strN)), nil
}

func (d *Device) getInfoUint(param C.cl_device_info, panicOnError bool) (uint, error) {
	var val C.cl_uint
	if err := C.clGetDeviceInfo(d.id, param, C.size_t(unsafe.Sizeof(val)), unsafe.Pointer(&val), nil); err != C.CL_SUCCESS {
		if panicOnError {
			panic("Should never fail")
		}
		return 0, CLError(err)
	}
	return uint(val), nil
}

func (d *Device) getInfoSize(param C.cl_device_info, panicOnError bool) (int, error) {
	var val C.size_t
	if err := C.clGetDeviceInfo(d.id, param, C.size_t(unsafe.Sizeof(val)), unsafe.Pointer(&val), nil); err != C.CL_SUCCESS {
		if panicOnError {
			panic("Should never fail")
		}
		return 0, CLError(err)
	}
	return int(val), nil
}

func (d *Device) getInfoBool(param C.cl_device_info, panicOnError bool) (bool, error) {
	var val C.cl_bool
	if err := C.clGetDeviceInfo(d.id, param, C.size_t(unsafe.Sizeof(val)), unsafe.Pointer(&val), nil); err != C.CL_SUCCESS {
		if panicOnError {
			panic("Should never fail")
		}
		return false, CLError(err)
	}
	return val == C.CL_TRUE, nil
}

func (d *Device) Name() string {
	str, _ := d.getInfoString(C.CL_DEVICE_NAME, true)
	return str
}

func (d *Device) Vendor() string {
	str, _ := d.getInfoString(C.CL_DEVICE_VENDOR, true)
	return str
}

func (d *Device) BuiltInKernels() string {
	str, _ := d.getInfoString(C.CL_DEVICE_BUILT_IN_KERNELS, true)
	return str
}

func (d *Device) Extensions() string {
	str, _ := d.getInfoString(C.CL_DEVICE_EXTENSIONS, true)
	return str
}

func (d *Device) OpenCLCVersion() string {
	str, _ := d.getInfoString(C.CL_DEVICE_OPENCL_C_VERSION, true)
	return str
}

func (d *Device) Profile() string {
	str, _ := d.getInfoString(C.CL_DEVICE_PROFILE, true)
	return str
}

func (d *Device) Version() string {
	str, _ := d.getInfoString(C.CL_DEVICE_VERSION, true)
	return str
}

func (d *Device) DriverVersion() string {
	str, _ := d.getInfoString(C.CL_DRIVER_VERSION, true)
	return str
}

// The default compute device address space size specified as an
// unsigned integer value in bits. Currently supported values are 32 or 64 bits.
func (d *Device) AddressBits() int {
	val, _ := d.getInfoUint(C.CL_DEVICE_ADDRESS_BITS, true)
	return int(val)
}

// Size of global memory cache line in bytes.
func (d *Device) GlobalMemCachelineSize() int {
	val, _ := d.getInfoUint(C.CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, true)
	return int(val)
}

// Maximum configured clock frequency of the device in MHz.
func (d *Device) MaxClockFrequency() int {
	val, _ := d.getInfoUint(C.CL_DEVICE_MAX_CLOCK_FREQUENCY, true)
	return int(val)
}

// The number of parallel compute units on the OpenCL device.
// A work-group executes on a single compute unit. The minimum value is 1.
func (d *Device) MaxComputeUnits() int {
	val, _ := d.getInfoUint(C.CL_DEVICE_MAX_COMPUTE_UNITS, true)
	return int(val)
}

// Max number of arguments declared with the __constant qualifier in a kernel.
// The minimum value is 8 for devices that are not of type CL_DEVICE_TYPE_CUSTOM.
func (d *Device) MaxConstantArgs() int {
	val, _ := d.getInfoUint(C.CL_DEVICE_MAX_CONSTANT_ARGS, true)
	return int(val)
}

// Max number of simultaneous image objects that can be read by a kernel.
// The minimum value is 128 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
func (d *Device) MaxReadImageArgs() int {
	val, _ := d.getInfoUint(C.CL_DEVICE_MAX_READ_IMAGE_ARGS, true)
	return int(val)
}

// Maximum number of samplers that can be used in a kernel. The minimum
// value is 16 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE. (Also see sampler_t.)
func (d *Device) MaxSamplers() int {
	val, _ := d.getInfoUint(C.CL_DEVICE_MAX_SAMPLERS, true)
	return int(val)
}

// Maximum dimensions that specify the global and local work-item IDs used
// by the data parallel execution model. (Refer to clEnqueueNDRangeKernel).
// The minimum value is 3 for devices that are not of type CL_DEVICE_TYPE_CUSTOM.
func (d *Device) MaxWorkItemDimensions() int {
	val, _ := d.getInfoUint(C.CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, true)
	return int(val)
}

// Max number of simultaneous image objects that can be written to by a
// kernel. The minimum value is 8 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
func (d *Device) MaxWriteImageArgs() int {
	val, _ := d.getInfoUint(C.CL_DEVICE_MAX_WRITE_IMAGE_ARGS, true)
	return int(val)
}

// The minimum value is the size (in bits) of the largest OpenCL built-in
// data type supported by the device (long16 in FULL profile, long16 or
// int16 in EMBEDDED profile) for devices that are not of type CL_DEVICE_TYPE_CUSTOM.
func (d *Device) MemBaseAddrAlign() int {
	val, _ := d.getInfoUint(C.CL_DEVICE_MEM_BASE_ADDR_ALIGN, true)
	return int(val)
}

// Max height of 2D image in pixels. The minimum value is 8192
// if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
func (d *Device) Image2DMaxHeight() int {
	val, _ := d.getInfoSize(C.CL_DEVICE_IMAGE2D_MAX_HEIGHT, true)
	return int(val)
}

// Max width of 2D image or 1D image not created from a buffer object in
// pixels. The minimum value is 8192 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
func (d *Device) Image2DMaxWidth() int {
	val, _ := d.getInfoSize(C.CL_DEVICE_IMAGE2D_MAX_WIDTH, true)
	return int(val)
}

// Max size in bytes of the arguments that can be passed to a kernel. The
// minimum value is 1024 for devices that are not of type CL_DEVICE_TYPE_CUSTOM.
// For this minimum value, only a maximum of 128 arguments can be passed to a kernel.
func (d *Device) MaxParameterSize() int {
	val, _ := d.getInfoSize(C.CL_DEVICE_MAX_PARAMETER_SIZE, true)
	return int(val)
}

// Maximum number of work-items in a work-group executing a kernel on a
// single compute unit, using the data parallel execution model. (Refer
// to clEnqueueNDRangeKernel). The minimum value is 1.
func (d *Device) MaxWorkGroupSize() int {
	val, _ := d.getInfoSize(C.CL_DEVICE_MAX_WORK_GROUP_SIZE, true)
	return int(val)
}

func (d *Device) Available() bool {
	val, _ := d.getInfoBool(C.CL_DEVICE_AVAILABLE, true)
	return val
}

func (d *Device) CompilerAvailable() bool {
	val, _ := d.getInfoBool(C.CL_DEVICE_COMPILER_AVAILABLE, true)
	return val
}

func (d *Device) EndianLittle() bool {
	val, _ := d.getInfoBool(C.CL_DEVICE_ENDIAN_LITTLE, true)
	return val
}

func (d *Device) ErrorCorrectionSupport() bool {
	val, _ := d.getInfoBool(C.CL_DEVICE_ERROR_CORRECTION_SUPPORT, true)
	return val
}

func (d *Device) HostUnifiedMemory() bool {
	val, _ := d.getInfoBool(C.CL_DEVICE_HOST_UNIFIED_MEMORY, true)
	return val
}

func (d *Device) ImageSupport() bool {
	val, _ := d.getInfoBool(C.CL_DEVICE_IMAGE_SUPPORT, true)
	return val
}

func (d *Device) Type() DeviceType {
	var deviceType C.cl_device_type
	if err := C.clGetDeviceInfo(d.id, C.CL_DEVICE_TYPE, C.size_t(unsafe.Sizeof(deviceType)), unsafe.Pointer(&deviceType), nil); err != C.CL_SUCCESS {
		panic("Failed to get device type")
	}
	return DeviceType(deviceType)
}

// Describes double precision floating-point capability of the OpenCL device
func (d *Device) DoubleFPConfig() FPConfig {
	var fpConfig C.cl_device_fp_config
	if err := C.clGetDeviceInfo(d.id, C.CL_DEVICE_DOUBLE_FP_CONFIG, C.size_t(unsafe.Sizeof(fpConfig)), unsafe.Pointer(&fpConfig), nil); err != C.CL_SUCCESS {
		panic("Failed to get double FP config")
	}
	return FPConfig(fpConfig)
}

// Describes the OPTIONAL half precision floating-point capability of the OpenCL device
func (d *Device) HalfFPConfig() FPConfig {
	var fpConfig C.cl_device_fp_config
	if err := C.clGetDeviceInfo(d.id, C.CL_DEVICE_HALF_FP_CONFIG, C.size_t(unsafe.Sizeof(fpConfig)), unsafe.Pointer(&fpConfig), nil); err != C.CL_SUCCESS {
		return FPConfig(0)
	}
	return FPConfig(fpConfig)
}
