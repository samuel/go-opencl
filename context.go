package cl

// #include <OpenCL/opencl.h>
import "C"

import (
	"image"
	"runtime"
	"unsafe"
)

const maxImageFormats = 256

type MemFlag int

const (
	MemReadWrite     MemFlag = C.CL_MEM_READ_WRITE
	MemWriteOnly     MemFlag = C.CL_MEM_WRITE_ONLY
	MemReadOnly      MemFlag = C.CL_MEM_READ_ONLY
	MemUseHostPtr    MemFlag = C.CL_MEM_USE_HOST_PTR
	MemAllocHostPtr  MemFlag = C.CL_MEM_ALLOC_HOST_PTR
	MemCopyHostPtr   MemFlag = C.CL_MEM_COPY_HOST_PTR
	MemHostWriteOnly MemFlag = C.CL_MEM_HOST_WRITE_ONLY // OpenCL 1.2
	MemHostReadOnly  MemFlag = C.CL_MEM_HOST_READ_ONLY  // OpenCL 1.2
	MemHostNoAccess  MemFlag = C.CL_MEM_HOST_NO_ACCESS  // OpenCL 1.2
)

type MemObjectType int

const (
	MemObjectTypeBuffer        MemObjectType = C.CL_MEM_OBJECT_BUFFER
	MemObjectTypeImage2D       MemObjectType = C.CL_MEM_OBJECT_IMAGE2D
	MemObjectTypeImage3D       MemObjectType = C.CL_MEM_OBJECT_IMAGE3D
	MemObjectTypeImage2DArray  MemObjectType = C.CL_MEM_OBJECT_IMAGE2D_ARRAY
	MemObjectTypeImage1D       MemObjectType = C.CL_MEM_OBJECT_IMAGE1D
	MemObjectTypeImage1DArray  MemObjectType = C.CL_MEM_OBJECT_IMAGE1D_ARRAY
	MemObjectTypeImage1DBuffer MemObjectType = C.CL_MEM_OBJECT_IMAGE1D_BUFFER
)

type ChannelOrder int

const (
	ChannelOrderR            ChannelOrder = C.CL_R
	ChannelOrderA            ChannelOrder = C.CL_A
	ChannelOrderRG           ChannelOrder = C.CL_RG
	ChannelOrderRA           ChannelOrder = C.CL_RA
	ChannelOrderRGB          ChannelOrder = C.CL_RGB
	ChannelOrderRGBA         ChannelOrder = C.CL_RGBA
	ChannelOrderBGRA         ChannelOrder = C.CL_BGRA
	ChannelOrderARGB         ChannelOrder = C.CL_ARGB
	ChannelOrderIntensity    ChannelOrder = C.CL_INTENSITY
	ChannelOrderLuminance    ChannelOrder = C.CL_LUMINANCE
	ChannelOrderRx           ChannelOrder = C.CL_Rx
	ChannelOrderRGx          ChannelOrder = C.CL_RGx
	ChannelOrderRGBx         ChannelOrder = C.CL_RGBx
	ChannelOrderDepth        ChannelOrder = C.CL_DEPTH
	ChannelOrderDepthStencil ChannelOrder = C.CL_DEPTH_STENCIL
)

var channelOrderNameMap = map[ChannelOrder]string{
	ChannelOrderR:            "R",
	ChannelOrderA:            "A",
	ChannelOrderRG:           "RG",
	ChannelOrderRA:           "RA",
	ChannelOrderRGB:          "RGB",
	ChannelOrderRGBA:         "RGBA",
	ChannelOrderBGRA:         "BGRA",
	ChannelOrderARGB:         "ARGB",
	ChannelOrderIntensity:    "Intensity",
	ChannelOrderLuminance:    "Luminance",
	ChannelOrderRx:           "Rx",
	ChannelOrderRGx:          "RGx",
	ChannelOrderRGBx:         "RGBx",
	ChannelOrderDepth:        "Depth",
	ChannelOrderDepthStencil: "DepthStencil",
}

func (co ChannelOrder) String() string {
	name := channelOrderNameMap[co]
	if name == "" {
		name = "Unknown"
	}
	return name
}

type ChannelDataType int

const (
	ChannelDataTypeSNormInt8      ChannelDataType = C.CL_SNORM_INT8
	ChannelDataTypeSNormInt16     ChannelDataType = C.CL_SNORM_INT16
	ChannelDataTypeUNormInt8      ChannelDataType = C.CL_UNORM_INT8
	ChannelDataTypeUNormInt16     ChannelDataType = C.CL_UNORM_INT16
	ChannelDataTypeUNormShort565  ChannelDataType = C.CL_UNORM_SHORT_565
	ChannelDataTypeUNormShort555  ChannelDataType = C.CL_UNORM_SHORT_555
	ChannelDataTypeUNormInt101010 ChannelDataType = C.CL_UNORM_INT_101010
	ChannelDataTypeSignedInt8     ChannelDataType = C.CL_SIGNED_INT8
	ChannelDataTypeSignedInt16    ChannelDataType = C.CL_SIGNED_INT16
	ChannelDataTypeSignedInt32    ChannelDataType = C.CL_SIGNED_INT32
	ChannelDataTypeUnsignedInt8   ChannelDataType = C.CL_UNSIGNED_INT8
	ChannelDataTypeUnsignedInt16  ChannelDataType = C.CL_UNSIGNED_INT16
	ChannelDataTypeUnsignedInt32  ChannelDataType = C.CL_UNSIGNED_INT32
	ChannelDataTypeHalfFloat      ChannelDataType = C.CL_HALF_FLOAT
	ChannelDataTypeFloat          ChannelDataType = C.CL_FLOAT
	ChannelDataTypeUNormInt24     ChannelDataType = C.CL_UNORM_INT24
)

var channelDataTypeNameMap = map[ChannelDataType]string{
	ChannelDataTypeSNormInt8:      "SNormInt8",
	ChannelDataTypeSNormInt16:     "SNormInt16",
	ChannelDataTypeUNormInt8:      "UNormInt8",
	ChannelDataTypeUNormInt16:     "UNormInt16",
	ChannelDataTypeUNormShort565:  "UNormShort565",
	ChannelDataTypeUNormShort555:  "UNormShort555",
	ChannelDataTypeUNormInt101010: "UNormInt101010",
	ChannelDataTypeSignedInt8:     "SignedInt8",
	ChannelDataTypeSignedInt16:    "SignedInt16",
	ChannelDataTypeSignedInt32:    "SignedInt32",
	ChannelDataTypeUnsignedInt8:   "UnsignedInt8",
	ChannelDataTypeUnsignedInt16:  "UnsignedInt16",
	ChannelDataTypeUnsignedInt32:  "UnsignedInt32",
	ChannelDataTypeHalfFloat:      "HalfFloat",
	ChannelDataTypeFloat:          "Float",
	ChannelDataTypeUNormInt24:     "UNormInt24",
}

func (ct ChannelDataType) String() string {
	name := channelDataTypeNameMap[ct]
	if name == "" {
		name = "Unknown"
	}
	return name
}

type ImageFormat struct {
	ChannelOrder    ChannelOrder
	ChannelDataType ChannelDataType
}

func (f ImageFormat) toCl() C.cl_image_format {
	var format C.cl_image_format
	format.image_channel_order = C.cl_channel_order(f.ChannelOrder)
	format.image_channel_data_type = C.cl_channel_type(f.ChannelDataType)
	return format
}

type ImageDescription struct {
	Type                            MemObjectType
	Width, Height, Depth            int
	ArraySize, RowPitch, SlicePitch int
	NumMipLevels, NumSamples        int
	Buffer                          *Buffer
}

func (d ImageDescription) toCl() C.cl_image_desc {
	var desc C.cl_image_desc
	desc.image_type = C.cl_mem_object_type(d.Type)
	desc.image_width = C.size_t(d.Width)
	desc.image_height = C.size_t(d.Height)
	desc.image_depth = C.size_t(d.Depth)
	desc.image_array_size = C.size_t(d.ArraySize)
	desc.image_row_pitch = C.size_t(d.RowPitch)
	desc.image_slice_pitch = C.size_t(d.SlicePitch)
	desc.num_mip_levels = C.cl_uint(d.NumMipLevels)
	desc.num_samples = C.cl_uint(d.NumSamples)
	desc.buffer = nil
	if d.Buffer != nil {
		desc.buffer = d.Buffer.clBuffer
	}
	return desc
}

type Context struct {
	clContext C.cl_context
	devices   []*Device
}

type Buffer struct {
	clBuffer C.cl_mem
	size     int
}

func releaseContext(c *Context) {
	if c.clContext != nil {
		C.clReleaseContext(c.clContext)
		c.clContext = nil
	}
}

func releaseBuffer(b *Buffer) {
	if b.clBuffer != nil {
		C.clReleaseMemObject(b.clBuffer)
		b.clBuffer = nil
	}
}

// TODO: properties
func CreateContext(devices []*Device) (*Context, error) {
	deviceIds := buildDeviceIdList(devices)
	var err C.cl_int
	clContext := C.clCreateContext(nil, C.cl_uint(len(devices)), &deviceIds[0], nil, nil, &err)
	if err != C.CL_SUCCESS {
		return nil, CLError(err)
	}
	if clContext == nil {
		return nil, ErrUnknown
	}
	context := &Context{clContext: clContext, devices: devices}
	runtime.SetFinalizer(context, releaseContext)
	return context, nil
}

func (ctx *Context) GetSupportedImageFormats(flags MemFlag, imageType MemObjectType) ([]ImageFormat, error) {
	var formats [maxImageFormats]C.cl_image_format
	var nFormats C.cl_uint
	if err := C.clGetSupportedImageFormats(ctx.clContext, C.cl_mem_flags(flags), C.cl_mem_object_type(imageType), maxImageFormats, &formats[0], &nFormats); err != C.CL_SUCCESS {
		return nil, CLError(err)
	}
	fmts := make([]ImageFormat, nFormats)
	for i, f := range formats[:nFormats] {
		fmts[i] = ImageFormat{
			ChannelOrder:    ChannelOrder(f.image_channel_order),
			ChannelDataType: ChannelDataType(f.image_channel_data_type),
		}
	}
	return fmts, nil
}

func (ctx *Context) CreateCommandQueue(device *Device, properties CommandQueueProperty) (*CommandQueue, error) {
	var err C.cl_int
	clQueue := C.clCreateCommandQueue(ctx.clContext, device.id, C.cl_command_queue_properties(properties), &err)
	if err != C.CL_SUCCESS {
		return nil, CLError(err)
	}
	if clQueue == nil {
		return nil, ErrUnknown
	}
	commandQueue := &CommandQueue{clQueue: clQueue, device: device}
	runtime.SetFinalizer(commandQueue, releaseCommandQueue)
	return commandQueue, nil
}

func (ctx *Context) CreateProgramWithSource(sources []string) (*Program, error) {
	cSources := make([]*C.char, len(sources))
	for i, s := range sources {
		cs := C.CString(s)
		cSources[i] = cs
		defer C.free(unsafe.Pointer(cs))
	}
	var err C.cl_int
	clProgram := C.clCreateProgramWithSource(ctx.clContext, C.cl_uint(len(sources)), &cSources[0], nil, &err)
	if err != C.CL_SUCCESS {
		return nil, CLError(err)
	}
	if clProgram == nil {
		return nil, ErrUnknown
	}
	program := &Program{clProgram: clProgram, devices: ctx.devices}
	runtime.SetFinalizer(program, releaseProgram)
	return program, nil
}

func (ctx *Context) CreateBuffer(flags MemFlag, size int, dataPtr unsafe.Pointer) (*Buffer, error) {
	var err C.cl_int
	clBuffer := C.clCreateBuffer(ctx.clContext, C.cl_mem_flags(flags), C.size_t(size), dataPtr, &err)
	if err != C.CL_SUCCESS {
		return nil, CLError(err)
	}
	if clBuffer == nil {
		return nil, ErrUnknown
	}
	buffer := &Buffer{clBuffer: clBuffer, size: size}
	runtime.SetFinalizer(buffer, releaseBuffer)
	return buffer, nil
}

// OpenCL 1.2
func (ctx *Context) CreateImage(flags MemFlag, imageFormat ImageFormat, imageDesc ImageDescription, data []byte) (*Buffer, error) {
	format := imageFormat.toCl()
	desc := imageDesc.toCl()
	var dataPtr unsafe.Pointer
	if data != nil {
		dataPtr = unsafe.Pointer(&data[0])
	}
	var err C.cl_int
	clBuffer := C.clCreateImage(ctx.clContext, C.cl_mem_flags(flags), &format, &desc, dataPtr, &err)
	if err != C.CL_SUCCESS {
		return nil, CLError(err)
	}
	if clBuffer == nil {
		return nil, ErrUnknown
	}
	buffer := &Buffer{clBuffer: clBuffer, size: -1}
	runtime.SetFinalizer(buffer, releaseBuffer)
	return buffer, nil
}

func (ctx *Context) CreateImageSimple(flags MemFlag, width, height int, channelOrder ChannelOrder, channelDataType ChannelDataType, data []byte) (*Buffer, error) {
	format := ImageFormat{channelOrder, channelDataType}
	desc := ImageDescription{
		Type:   MemObjectTypeImage2D,
		Width:  width,
		Height: height,
	}
	return ctx.CreateImage(flags, format, desc, data)
}

func (ctx *Context) CreateImageFromImage(flags MemFlag, img image.Image) (*Buffer, error) {
	switch m := img.(type) {
	case *image.Gray:
		format := ImageFormat{ChannelOrderIntensity, ChannelDataTypeUNormInt8}
		desc := ImageDescription{
			Type:     MemObjectTypeImage2D,
			Width:    m.Bounds().Dx(),
			Height:   m.Bounds().Dy(),
			RowPitch: m.Stride,
		}
		return ctx.CreateImage(flags, format, desc, m.Pix)
	case *image.RGBA:
		format := ImageFormat{ChannelOrderRGBA, ChannelDataTypeUNormInt8}
		desc := ImageDescription{
			Type:     MemObjectTypeImage2D,
			Width:    m.Bounds().Dx(),
			Height:   m.Bounds().Dy(),
			RowPitch: m.Stride,
		}
		return ctx.CreateImage(flags, format, desc, m.Pix)
	}

	b := img.Bounds()
	w := b.Dx()
	h := b.Dy()
	data := make([]byte, w*h*4)
	dataOffset := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img.At(x+b.Min.X, y+b.Min.Y)
			r, g, b, a := c.RGBA()
			data[dataOffset] = uint8(r >> 8)
			data[dataOffset+1] = uint8(g >> 8)
			data[dataOffset+2] = uint8(b >> 8)
			data[dataOffset+3] = uint8(a >> 8)
			dataOffset += 4
		}
	}
	return ctx.CreateImageSimple(flags, w, h, ChannelOrderRGBA, ChannelDataTypeUNormInt8, data)
}
