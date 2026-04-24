package rustbridge

/*
#cgo LDFLAGS: -lipp_ffi
#include "ipp_ffi.h"
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"encoding/json"
	"fmt"
	"unsafe"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	RustPluginType = "rust-bridge"
)

// RustPluginChain wraps the Rust FFI plugin chain as a single BBR plugin.
// It implements both RequestProcessor and ResponseProcessor.
type RustPluginChain struct {
	typedName plugin.TypedName
	chain     *C.struct_IppChain
}

// NewRustPluginChain initializes the Rust plugin chain via FFI.
// vertexConfig is optional JSON with project/location/endpoint for Vertex OpenAI.
func NewRustPluginChain(vertexConfig string) (*RustPluginChain, error) {
	var configPtr *C.char
	if vertexConfig != "" {
		configPtr = C.CString(vertexConfig)
		defer C.free(unsafe.Pointer(configPtr))
	}

	chain := C.ipp_init(configPtr)
	if chain == nil {
		return nil, fmt.Errorf("failed to initialize Rust plugin chain")
	}

	return &RustPluginChain{
		typedName: plugin.TypedName{
			Type: RustPluginType,
			Name: RustPluginType,
		},
		chain: chain,
	}, nil
}

func (p *RustPluginChain) TypedName() plugin.TypedName {
	return p.typedName
}

func (p *RustPluginChain) WithName(name string) *RustPluginChain {
	p.typedName.Name = name
	return p
}

// ProcessRequest marshals the request headers and body to C, calls the Rust
// plugin chain, and applies the returned mutations to the Go InferenceRequest.
func (p *RustPluginChain) ProcessRequest(ctx context.Context, cycleState *framework.CycleState, request *framework.InferenceRequest) error {
	headersJSON, err := json.Marshal(request.Headers)
	if err != nil {
		return fmt.Errorf("failed to marshal headers: %w", err)
	}

	bodyBytes, err := json.Marshal(request.Body)
	if err != nil {
		return fmt.Errorf("failed to marshal body: %w", err)
	}

	cHeaders := C.CString(string(headersJSON))
	defer C.free(unsafe.Pointer(cHeaders))

	cBody := C.CBytes(bodyBytes)
	defer C.free(cBody)

	result := C.ipp_process_request(p.chain, cHeaders, (*C.uint8_t)(cBody), C.uintptr_t(len(bodyBytes)))
	if result == nil {
		return fmt.Errorf("ipp_process_request returned nil")
	}
	defer C.ipp_free_result(result)

	if result.error_code != 0 {
		errMsg := C.GoString(result.error_msg)
		return fmt.Errorf("rust plugin error (%d): %s", result.error_code, errMsg)
	}

	// Apply header mutations
	if result.mutated_headers_json != nil {
		var mutatedHeaders map[string]string
		if err := json.Unmarshal([]byte(C.GoString(result.mutated_headers_json)), &mutatedHeaders); err == nil {
			for k, v := range mutatedHeaders {
				request.SetHeader(k, v)
			}
		}
	}

	if result.removed_headers_json != nil {
		var removedHeaders []string
		if err := json.Unmarshal([]byte(C.GoString(result.removed_headers_json)), &removedHeaders); err == nil {
			for _, k := range removedHeaders {
				request.RemoveHeader(k)
			}
		}
	}

	// Apply body mutation
	if result.mutated_body != nil && result.mutated_body_len > 0 {
		bodySlice := C.GoBytes(unsafe.Pointer(result.mutated_body), C.int(result.mutated_body_len))
		var newBody map[string]any
		if err := json.Unmarshal(bodySlice, &newBody); err == nil {
			request.SetBody(newBody)
		}
	}

	return nil
}

// ProcessResponse marshals the response headers and body to C, calls the Rust
// plugin chain, and applies the returned mutations to the Go InferenceResponse.
func (p *RustPluginChain) ProcessResponse(ctx context.Context, cycleState *framework.CycleState, response *framework.InferenceResponse) error {
	headersJSON, err := json.Marshal(response.Headers)
	if err != nil {
		return fmt.Errorf("failed to marshal headers: %w", err)
	}

	bodyBytes, err := json.Marshal(response.Body)
	if err != nil {
		return fmt.Errorf("failed to marshal body: %w", err)
	}

	cHeaders := C.CString(string(headersJSON))
	defer C.free(unsafe.Pointer(cHeaders))

	cBody := C.CBytes(bodyBytes)
	defer C.free(cBody)

	result := C.ipp_process_response(p.chain, cHeaders, (*C.uint8_t)(cBody), C.uintptr_t(len(bodyBytes)))
	if result == nil {
		return fmt.Errorf("ipp_process_response returned nil")
	}
	defer C.ipp_free_result(result)

	if result.error_code != 0 {
		errMsg := C.GoString(result.error_msg)
		return fmt.Errorf("rust plugin error (%d): %s", result.error_code, errMsg)
	}

	// Apply header mutations
	if result.mutated_headers_json != nil {
		var mutatedHeaders map[string]string
		if err := json.Unmarshal([]byte(C.GoString(result.mutated_headers_json)), &mutatedHeaders); err == nil {
			for k, v := range mutatedHeaders {
				response.SetHeader(k, v)
			}
		}
	}

	if result.removed_headers_json != nil {
		var removedHeaders []string
		if err := json.Unmarshal([]byte(C.GoString(result.removed_headers_json)), &removedHeaders); err == nil {
			for _, k := range removedHeaders {
				response.RemoveHeader(k)
			}
		}
	}

	// Apply body mutation
	if result.mutated_body != nil && result.mutated_body_len > 0 {
		bodySlice := C.GoBytes(unsafe.Pointer(result.mutated_body), C.int(result.mutated_body_len))
		var newBody map[string]any
		if err := json.Unmarshal(bodySlice, &newBody); err == nil {
			response.SetBody(newBody)
		}
	}

	return nil
}

// Destroy shuts down the Rust plugin chain and tokio runtime.
func (p *RustPluginChain) Destroy() {
	if p.chain != nil {
		C.ipp_destroy(p.chain)
		p.chain = nil
	}
}
