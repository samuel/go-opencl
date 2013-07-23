/*
Package opencl provides a binding to the OpenCL api. It's mostly a low-level
wrapper that avoids adding functionality while still making the interface
a little more friendly and easy to use.
*/
package opencl

import (
	_ "github.com/dennwc/opencl/cl11"
	_ "github.com/dennwc/opencl/cl12"
)
