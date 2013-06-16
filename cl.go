package cl

// #include <OpenCL/opencl.h>
// #cgo LDFLAGS: -framework OpenCL -Qunused-arguments
import "C"

import (
	"errors"
	"fmt"
)

var ErrUnknown = errors.New("cl: unknown error")

type CLError int

func (e CLError) Error() string {
	return fmt.Sprintf("cl: error %d", int(e))
}
