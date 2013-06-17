package cl

// #include <OpenCL/opencl.h>
import "C"

import (
	"image"
	"runtime"
	"unsafe"
)

const maxImageFormats = 256

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
		return nil, toError(err)
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
		return nil, toError(err)
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
		return nil, toError(err)
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
		return nil, toError(err)
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
		return nil, toError(err)
	}
	if clBuffer == nil {
		return nil, ErrUnknown
	}
	buffer := &Buffer{clBuffer: clBuffer, size: size}
	runtime.SetFinalizer(buffer, releaseBuffer)
	return buffer, nil
}

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
		return nil, toError(err)
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
