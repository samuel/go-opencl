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

func (q *CommandQueue) validate() {
	if q.clQueue == nil {
		panic("cl: CommandQueue is nil")
	}
}

// Call clReleaseCommandQueue on the CommandQueue. Using the CommandQueue after Release will cause a panick.
func (q *CommandQueue) Release() {
	releaseCommandQueue(q)
}

// Blocks until all previously queued OpenCL commands in a command-queue are issued to the associated device and have completed.
func (q *CommandQueue) Finish() error {
	q.validate()
	return toError(C.clFinish(q.clQueue))
}

// Issues all previously queued OpenCL commands in a command-queue to the device associated with the command-queue.
func (q *CommandQueue) Flush() error {
	q.validate()
	return toError(C.clFinish(q.clQueue))
}

// Enqueues a command to copy a buffer object to another buffer object.
func (q *CommandQueue) EnqueueCopyBuffer(srcBuffer, dstBuffer *Buffer, srcOffset, dstOffset, byteCount int, eventWaitList []Event) (Event, error) {
	q.validate()
	srcBuffer.validate()
	dstBuffer.validate()
	var event C.cl_event
	err := toError(C.clEnqueueCopyBuffer(q.clQueue, srcBuffer.clBuffer, dstBuffer.clBuffer, C.size_t(srcOffset), C.size_t(dstOffset), C.size_t(byteCount), C.cl_uint(len(eventWaitList)), eventListPtr(eventWaitList), &event))
	return Event(event), err
}

// Enqueue commands to write to a buffer object from host memory.
func (q *CommandQueue) EnqueueWriteBuffer(buffer *Buffer, blocking bool, offset, dataSize int, dataPtr unsafe.Pointer, eventWaitList []Event) (Event, error) {
	q.validate()
	buffer.validate()
	var event C.cl_event
	err := toError(C.clEnqueueWriteBuffer(q.clQueue, buffer.clBuffer, clBool(blocking), C.size_t(offset), C.size_t(dataSize), dataPtr, C.cl_uint(len(eventWaitList)), eventListPtr(eventWaitList), &event))
	return Event(event), err
}

func (q *CommandQueue) EnqueueWriteBufferFloat32(buffer *Buffer, blocking bool, offset int, data []float32, eventWaitList []Event) (Event, error) {
	dataPtr := unsafe.Pointer(&data[0])
	dataSize := int(unsafe.Sizeof(data[0])) * len(data)
	return q.EnqueueWriteBuffer(buffer, blocking, offset, dataSize, dataPtr, eventWaitList)
}

// Enqueue commands to read from a buffer object to host memory.
func (q *CommandQueue) EnqueueReadBuffer(buffer *Buffer, blocking bool, offset, dataSize int, dataPtr unsafe.Pointer, eventWaitList []Event) (Event, error) {
	q.validate()
	buffer.validate()
	var event C.cl_event
	err := toError(C.clEnqueueReadBuffer(q.clQueue, buffer.clBuffer, clBool(blocking), C.size_t(offset), C.size_t(dataSize), dataPtr, C.cl_uint(len(eventWaitList)), eventListPtr(eventWaitList), &event))
	return Event(event), err
}

func (q *CommandQueue) EnqueueReadBufferFloat32(buffer *Buffer, blocking bool, offset int, data []float32, eventWaitList []Event) (Event, error) {
	dataPtr := unsafe.Pointer(&data[0])
	dataSize := int(unsafe.Sizeof(data[0])) * len(data)
	return q.EnqueueReadBuffer(buffer, blocking, offset, dataSize, dataPtr, eventWaitList)
}

// Enqueues a command to fill a buffer object with a pattern of a given pattern size.
func (q *CommandQueue) EnqueueFillBuffer(buffer *Buffer, pattern unsafe.Pointer, patternSize, offset, size int, eventWaitList []Event) (Event, error) {
	q.validate()
	buffer.validate()
	var event C.cl_event
	err := toError(C.clEnqueueFillBuffer(q.clQueue, buffer.clBuffer, pattern, C.size_t(patternSize), C.size_t(offset), C.size_t(size), C.cl_uint(len(eventWaitList)), eventListPtr(eventWaitList), &event))
	return Event(event), err
}

// A synchronization point that enqueues a barrier operation.
func (q *CommandQueue) EnqueueBarrierWithWaitList(eventWaitList []Event) (Event, error) {
	q.validate()
	var event C.cl_event
	err := toError(C.clEnqueueBarrierWithWaitList(q.clQueue, C.cl_uint(len(eventWaitList)), eventListPtr(eventWaitList), &event))
	return Event(event), err
}

// Enqueues a command to execute a kernel on a device.
func (q *CommandQueue) EnqueueNDRangeKernel(kernel *Kernel, globalWorkOffset, globalWorkSize, localWorkSize []int, eventWaitList []Event) (Event, error) {
	q.validate()
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
	var event C.cl_event
	err := toError(C.clEnqueueNDRangeKernel(q.clQueue, kernel.clKernel, C.cl_uint(workDim), globalWorkOffsetPtr, globalWorkSizePtr, localWorkSizePtr, C.cl_uint(len(eventWaitList)), eventListPtr(eventWaitList), &event))
	return Event(event), err
}

// Enqueues a command to read from a 2D or 3D image object to host memory.
func (q *CommandQueue) EnqueueReadImage(image *Buffer, blocking bool, origin, region [3]int, rowPitch, slicePitch int, data []byte, eventWaitList []Event) (Event, error) {
	q.validate()
	image.validate()
	cOrigin := sizeT3(origin)
	cRegion := sizeT3(region)
	var event C.cl_event
	err := toError(C.clEnqueueReadImage(q.clQueue, image.clBuffer, clBool(blocking), &cOrigin[0], &cRegion[0], C.size_t(rowPitch), C.size_t(slicePitch), unsafe.Pointer(&data[0]), C.cl_uint(len(eventWaitList)), eventListPtr(eventWaitList), &event))
	return Event(event), err
}

// Enqueues a command to write from a 2D or 3D image object to host memory.
func (q *CommandQueue) EnqueueWriteImage(image *Buffer, blocking bool, origin, region [3]int, rowPitch, slicePitch int, data []byte, eventWaitList []Event) (Event, error) {
	q.validate()
	image.validate()
	cOrigin := sizeT3(origin)
	cRegion := sizeT3(region)
	var event C.cl_event
	err := toError(C.clEnqueueWriteImage(q.clQueue, image.clBuffer, clBool(blocking), &cOrigin[0], &cRegion[0], C.size_t(rowPitch), C.size_t(slicePitch), unsafe.Pointer(&data[0]), C.cl_uint(len(eventWaitList)), eventListPtr(eventWaitList), &event))
	return Event(event), err
}
