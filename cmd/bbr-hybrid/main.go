package main

import (
	"encoding/json"
	"fmt"
	"os"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/gateway-api-inference-extension/cmd/bbr/runner"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/rustbridge"
)

const (
	RustBridgePluginType = "rust-bridge"
)

func RustBridgeFactory(name string, rawConfig json.RawMessage, _ framework.Handle) (framework.BBRPlugin, error) {
	vertexConfig := ""
	if len(rawConfig) > 0 {
		var config struct {
			VertexOpenAI *struct {
				Project  string `json:"project"`
				Location string `json:"location"`
				Endpoint string `json:"endpoint"`
			} `json:"vertexOpenAI,omitempty"`
		}
		if err := json.Unmarshal(rawConfig, &config); err == nil && config.VertexOpenAI != nil {
			vBytes, _ := json.Marshal(config.VertexOpenAI)
			vertexConfig = string(vBytes)
		}
	}

	chain, err := rustbridge.NewRustPluginChain(vertexConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create rust-bridge plugin: %w", err)
	}
	return chain.WithName(name), nil
}

func main() {
	framework.Register(RustBridgePluginType, RustBridgeFactory)

	if err := runner.NewRunner().
		WithExecutableName("bbr-hybrid").
		Run(ctrl.SetupSignalHandler()); err != nil {
		os.Exit(1)
	}
}
