package main

import (
	"yoloonnx"
	"go.viam.com/rdk/module"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/mlmodel"
)

func main() {
	// ModularMain can take multiple APIModel arguments, if your module implements multiple models.
	module.ModularMain(resource.APIModel{ mlmodel.API, yoloonnx.YoloOnnxCpu})
}
