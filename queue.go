package cl

// #include <OpenCL/opencl.h>
import "C"

import (
	"unsafe"
)

type CommandQueueProperty int

const (
	QueueOutOfOrderExecModeEnable CommandQueueProperty = C.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
	QueueProfilingEnable          CommandQueueProperty = C.CL_QUEUE_PROFILING_ENABLE
)

type CommandQueue struct {
	clQueue C.cl_command_queue
	device  *Device
}

func releaseCommandQueue(q *CommandQueue) {
	if q.clQueue != nil {
		C.clReleaseCommandQueue(q.clQueue)
		q.clQueue = nil
	}
}

func (q *CommandQueue) Finish() error {
	if err := C.clFinish(q.clQueue); err != C.CL_SUCCESS {
		return CLError(err)
	}
	return nil
}

// TODO: event wait list
func (q *CommandQueue) EnqueueWriteBuffer(buffer *Buffer, blocking bool, offset, dataSize int, dataPtr unsafe.Pointer) error {
	cBlocking := C.cl_bool(C.CL_FALSE)
	if blocking {
		cBlocking = C.CL_TRUE
	}
	err := C.clEnqueueWriteBuffer(q.clQueue, buffer.clBuffer, cBlocking, C.size_t(offset), C.size_t(dataSize), dataPtr, 0, nil, nil)
	if err != C.CL_SUCCESS {
		return CLError(err)
	}
	return nil
}

func (q *CommandQueue) EnqueueWriteBufferFloat32(buffer *Buffer, blocking bool, offset int, data []float32) error {
	dataPtr := unsafe.Pointer(&data[0])
	dataSize := int(unsafe.Sizeof(data[0])) * len(data)
	return q.EnqueueWriteBuffer(buffer, blocking, offset, dataSize, dataPtr)
}

// TODO: event wait list
func (q *CommandQueue) EnqueueReadBuffer(buffer *Buffer, blocking bool, offset, dataSize int, dataPtr unsafe.Pointer) error {
	cBlocking := C.cl_bool(C.CL_FALSE)
	if blocking {
		cBlocking = C.CL_TRUE
	}
	err := C.clEnqueueReadBuffer(q.clQueue, buffer.clBuffer, cBlocking, C.size_t(offset), C.size_t(dataSize), dataPtr, 0, nil, nil)
	if err != C.CL_SUCCESS {
		return CLError(err)
	}
	return nil
}

func (q *CommandQueue) EnqueueReadBufferFloat32(buffer *Buffer, blocking bool, offset int, data []float32) error {
	dataPtr := unsafe.Pointer(&data[0])
	dataSize := int(unsafe.Sizeof(data[0])) * len(data)
	return q.EnqueueReadBuffer(buffer, blocking, offset, dataSize, dataPtr)
}

// TODO: event wait list
func (q *CommandQueue) EnqueueNDRangeKernel(kernel *Kernel, globalWorkOffset, globalWorkSize, localWorkSize []int) error {
	workDim := len(globalWorkSize)
	var globalWorkOffsetList []C.size_t
	var globalWorkOffsetPtr *C.size_t
	if globalWorkOffset != nil {
		globalWorkOffsetList = make([]C.size_t, len(globalWorkOffset))
		for i, off := range globalWorkOffset {
			globalWorkOffsetList[i] = C.size_t(off)
		}
		globalWorkOffsetPtr = &globalWorkOffsetList[0]
	}
	var globalWorkSizeList []C.size_t
	var globalWorkSizePtr *C.size_t
	if globalWorkSize != nil {
		globalWorkSizeList = make([]C.size_t, len(globalWorkSize))
		for i, off := range globalWorkSize {
			globalWorkSizeList[i] = C.size_t(off)
		}
		globalWorkSizePtr = &globalWorkSizeList[0]
	}
	var localWorkSizeList []C.size_t
	var localWorkSizePtr *C.size_t
	if localWorkSize != nil {
		localWorkSizeList = make([]C.size_t, len(localWorkSize))
		for i, off := range localWorkSize {
			localWorkSizeList[i] = C.size_t(off)
		}
		localWorkSizePtr = &localWorkSizeList[0]
	}
	err := C.clEnqueueNDRangeKernel(q.clQueue, kernel.clKernel, C.cl_uint(workDim), globalWorkOffsetPtr, globalWorkSizePtr, localWorkSizePtr, 0, nil, nil)
	if err != C.CL_SUCCESS {
		return CLError(err)
	}
	return nil
}

// TODO: event wait list
func (q *CommandQueue) EnqueueReadImage(image *Buffer, blocking bool, origin, region [3]int, rowPitch, slicePitch int, data []byte) error {
	cBlocking := C.cl_bool(C.CL_FALSE)
	if blocking {
		cBlocking = C.CL_TRUE
	}
	var cOrigin [3]C.size_t
	cOrigin[0] = C.size_t(origin[0])
	cOrigin[1] = C.size_t(origin[1])
	cOrigin[2] = C.size_t(origin[2])
	var cRegion [3]C.size_t
	cRegion[0] = C.size_t(region[0])
	cRegion[1] = C.size_t(region[1])
	cRegion[2] = C.size_t(region[2])
	err := C.clEnqueueReadImage(q.clQueue, image.clBuffer, cBlocking, &cOrigin[0], &cRegion[0], C.size_t(rowPitch), C.size_t(slicePitch), unsafe.Pointer(&data[0]), 0, nil, nil)
	if err != C.CL_SUCCESS {
		return CLError(err)
	}
	return nil
}

// TODO: event wait list
func (q *CommandQueue) EnqueueWriteImage(image *Buffer, blocking bool, origin, region [3]int, rowPitch, slicePitch int, data []byte) error {
	cBlocking := C.cl_bool(C.CL_FALSE)
	if blocking {
		cBlocking = C.CL_TRUE
	}
	var cOrigin [3]C.size_t
	cOrigin[0] = C.size_t(origin[0])
	cOrigin[1] = C.size_t(origin[1])
	cOrigin[2] = C.size_t(origin[2])
	var cRegion [3]C.size_t
	cRegion[0] = C.size_t(region[0])
	cRegion[1] = C.size_t(region[1])
	cRegion[2] = C.size_t(region[2])
	err := C.clEnqueueWriteImage(q.clQueue, image.clBuffer, cBlocking, &cOrigin[0], &cRegion[0], C.size_t(rowPitch), C.size_t(slicePitch), unsafe.Pointer(&data[0]), 0, nil, nil)
	if err != C.CL_SUCCESS {
		return CLError(err)
	}
	return nil
}
