package yoloonnx

import (
	"context"
	"fmt"
	"path"
	"runtime"
	"strings"

	"github.com/pkg/errors"
	ort "github.com/yalue/onnxruntime_go"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/ml"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/mlmodel"
	"go.viam.com/utils"
	"go.viam.com/utils/rpc"
	"gorgonia.org/tensor"
)

var (
	YoloOnnxCpu      = resource.NewModel("hipsterbrown", "yolo-onnx", "yolo-onnx-cpu")
	errUnimplemented = errors.New("unimplemented")
)

// DataTypeMap maps the long ONNX data type labels to the data type as written in Go.
var DataTypeMap = map[ort.TensorElementDataType]string{
	ort.TensorElementDataTypeFloat: "float32",
	ort.TensorElementDataTypeUint8: "uint8",
}

var blank []float32

func init() {
	resource.RegisterService(mlmodel.API, YoloOnnxCpu,
		resource.Registration[mlmodel.Service, *Config]{
			Constructor: newYoloOnnxYoloOnnxCpu,
		},
	)
}

type Config struct {
	/*
		Put config attributes here. There should be public/exported fields
		with a `json` parameter at the end of each attribute.

		Example config struct:
			type Config struct {
				Pin   string `json:"pin"`
				Board string `json:"board"`
				MinDeg *float64 `json:"min_angle_deg,omitempty"`
			}

		If your model does not need a config, replace *Config in the init
		function with resource.NoNativeConfig
	*/
	ModelPath string `json:"model_path"`
	LabelPath string `json:"label_path"`
}

type modelSession struct {
	Session *ort.AdvancedSession
	Input   *ort.Tensor[float32]
	Output  *ort.Tensor[float32]
}

// Validate ensures all parts of the config are valid and important fields exist.
// Returns implicit dependencies based on the config.
// The path is the JSON path in your robot's config (not the `Config` struct) to the
// resource being validated; e.g. "components.0".
func (cfg *Config) Validate(validatePath string) ([]string, error) {
	// Add config validation code here
	if cfg.ModelPath == "" {
		return nil, utils.NewConfigValidationFieldRequiredError(validatePath, "model_path")
	}
	ext := path.Ext(cfg.ModelPath)
	if ext != ".onnx" {
		base := path.Base(cfg.ModelPath)
		return nil, errors.Errorf("model_path filename must end in .onnx. The filename is %s", base)
	}
	return nil, nil
}

type yoloOnnxYoloOnnxCpu struct {
	resource.AlwaysRebuild

	name resource.Name

	logger logging.Logger
	cfg    *Config

	cancelCtx  context.Context
	cancelFunc func()

	session  modelSession
	metadata mlmodel.MLMetadata
}

func newYoloOnnxYoloOnnxCpu(ctx context.Context, deps resource.Dependencies, rawConf resource.Config, logger logging.Logger) (mlmodel.Service, error) {
	conf, err := resource.NativeConfig[*Config](rawConf)
	if err != nil {
		return nil, err
	}

	return NewYoloOnnxCpu(ctx, deps, rawConf.ResourceName(), conf, logger)

}

func NewYoloOnnxCpu(ctx context.Context, deps resource.Dependencies, name resource.Name, conf *Config, logger logging.Logger) (mlmodel.Service, error) {

	cancelCtx, cancelFunc := context.WithCancel(context.Background())

	yolo := &yoloOnnxYoloOnnxCpu{
		name:       name,
		logger:     logger,
		cfg:        conf,
		cancelCtx:  cancelCtx,
		cancelFunc: cancelFunc,
	}

	libPath, err := getSharedLibPath()
	if err != nil {
		return nil, err
	}
	ort.SetSharedLibraryPath(libPath)
	err = ort.InitializeEnvironment()
	if err != nil {
		return nil, err
	}

	_, outputInfo, err := ort.GetInputOutputInfo(conf.ModelPath)
	if err != nil {
		return nil, err
	}
	// create the metadata
	yolo.metadata = createMetadata(outputInfo, conf.LabelPath)
	// declare the input Tensor for the YOLO model
	// fill blank tensor with an "image" of the correct size
	// the image
	blank = make([]float32, 640*640*3)
	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewTensor(inputShape, blank)
	if err != nil {
		return nil, err
	}

	out0OrtTensorInfo := outputInfo[0]
	// detection_anchor_indices
	outputShape0 := ort.NewShape(out0OrtTensorInfo.Dimensions...)
	outputTensor0, err := ort.NewEmptyTensor[float32](outputShape0)
	if err != nil {
		return nil, err
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}

	session, err := ort.NewAdvancedSession(conf.ModelPath,
		[]string{"images"},
		[]string{"output0"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor0},
		options,
	)

	modelSes := modelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensor0,
	}
	yolo.session = modelSes
	return yolo, nil
}

func (yolo *yoloOnnxYoloOnnxCpu) Name() resource.Name {
	return yolo.name
}

func (yolo *yoloOnnxYoloOnnxCpu) NewClientFromConn(ctx context.Context, conn rpc.ClientConn, remoteName string, name resource.Name, logger logging.Logger) (mlmodel.Service, error) {
	return nil, resource.ErrDoUnimplemented
}

func (yolo *yoloOnnxYoloOnnxCpu) Infer(ctx context.Context, tensors ml.Tensors) (ml.Tensors, error) {
	input, err := processInput(tensors)
	if err != nil {
		return nil, err
	}
	inTensor := yolo.session.Input.GetData()
	copy(inTensor, input)
	err = yolo.session.Session.Run()
	if err != nil {
		return nil, err
	}
	out := yolo.session.Output.GetData()
	return yolo.processOutput(out)
}

func (yolo *yoloOnnxYoloOnnxCpu) Metadata(ctx context.Context) (mlmodel.MLMetadata, error) {
	return yolo.metadata, nil
}

func (yolo *yoloOnnxYoloOnnxCpu) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	return nil, resource.ErrDoUnimplemented
}

func (yolo *yoloOnnxYoloOnnxCpu) Close(context.Context) error {
	// Put close code here
	yolo.cancelFunc()
	// destroy session
	err := yolo.session.Session.Destroy()
	if err != nil {
		return err
	}
	err = yolo.session.Output.Destroy()
	if err != nil {
		return err
	}
	err = yolo.session.Input.Destroy()
	if err != nil {
		return err
	}
	// destroy environment
	err = ort.DestroyEnvironment()
	if err != nil {
		return err
	}
	return nil
}

func processInput(tensors ml.Tensors) ([]float32, error) {
	var imageTensor *tensor.Dense
	// if length of tensors is 1, just grab the first tensor
	// if more than 1 grab the one called input tensor, or image
	if len(tensors) == 1 {
		for _, t := range tensors {
			imageTensor = t
			break
		}
	} else {
		for name, t := range tensors {
			if name == "image" || name == "images" {
				imageTensor = t
				break
			}
		}
	}
	if imageTensor == nil {
		return nil, errors.New("no valid input tensor called 'image' or 'images' found")
	}
	if float32Data, ok := imageTensor.Data().([]float32); ok {
		return float32Data, nil
	}
	return nil, errors.Errorf("input tensor must be of tensor type UIn8, got %v", imageTensor.Dtype())
}

func (yolo *yoloOnnxYoloOnnxCpu) processOutput(output []float32) (ml.Tensors, error) {
	// there is 1 output tensor. Turn it into 3 tensors with the right backing
	outputShape := yolo.session.Output.GetShape()
	if len(outputShape) != 3 {
		return nil, fmt.Errorf("unexpected output shape for YOLO model: %v", outputShape)
	}
	yolo.logger.Info(len(output))
	numChannels := int(outputShape[1])
	numBoxes := int(outputShape[2])
	numClasses := numChannels - 4

	boxesData := make([]float32, numBoxes*4)
	scoresData := make([]float32, numBoxes)
	classProbsData := make([]float32, numBoxes)

	for i := range 4 {
		for j := range numBoxes {
			boxesData[j*4+i] = output[i*numBoxes+j]
		}
	}

	for i := range numBoxes {
		maxClassScore := float32(0)
		maxClassIndex := float32(0)

		for j := range numClasses {
			channelIdx := 4 + j

			if channelIdx < numChannels {
				score := output[channelIdx*numBoxes+i]
				if score > maxClassScore {
					maxClassScore = score
					maxClassIndex = float32(j)
				}
			}
		}

		scoresData[i] = maxClassScore
		classProbsData[i] = maxClassIndex
	}

	boxesShape := []int{1, numBoxes, 4}
	scoresShape := []int{1, numBoxes}
	classProbsShape := []int{1, numBoxes}
	outMap := ml.Tensors{}
	outMap["location"] = tensor.New(
		tensor.WithShape(boxesShape...),
		tensor.WithBacking(boxesData),
	)
	outMap["category"] = tensor.New(
		tensor.WithShape(classProbsShape...),
		tensor.WithBacking(classProbsData),
	)
	outMap["score"] = tensor.New(
		tensor.WithShape(scoresShape...),
		tensor.WithBacking(scoresData),
	)
	return outMap, nil
}

func getSharedLibPath() (string, error) {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "../third_party/onnxruntime.dll", nil
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "../third_party/onnxruntime_arm64.dylib", nil
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "../third_party/onnxruntime_arm64.so", nil
		}
		return "../third_party/onnxruntime.so", nil
	}
	switch arch := strings.Join([]string{runtime.GOOS, runtime.GOARCH}, "-"); arch {
	case "android-386":
		return "../third_party/onnx-android-x86.so", nil
	}
	return "", errors.Errorf("Unable to find a version of the onnxruntime library supporting %s %s", runtime.GOOS, runtime.GOARCH)
}

func createMetadata(outputInfo []ort.InputOutputInfo, labelPath string) mlmodel.MLMetadata {
	md := mlmodel.MLMetadata{}
	md.ModelName = "yolo_onnx"
	md.ModelDescription = "a pre-trained YOLO model in ONNX format used to detect or classify objects"
	// inputs
	inputs := []mlmodel.TensorInfo{}
	imageIn := mlmodel.TensorInfo{
		Name:     "images",
		DataType: "float32",
		Shape:    []int{1, 3, 480, 640},
	}
	inputs = append(inputs, imageIn)
	md.Inputs = inputs

	// outputs
	outputs := []mlmodel.TensorInfo{}
	extra := map[string]interface{}{}
	extra["labels"] = labelPath

	out0OrtTensorInfo := outputInfo[0]

	out0Shape := out0OrtTensorInfo.Dimensions
	numBoxes := int(out0Shape[2])

	boxesShape := []int{1, numBoxes, 4}
	scoresShape := []int{1, numBoxes}
	classProbsShape := []int{1, numBoxes}

	out1 := mlmodel.TensorInfo{
		Name:     "location",
		DataType: "float32",
		Shape:    boxesShape,
		Extra:    extra,
	}
	out2 := mlmodel.TensorInfo{
		Name:     "category",
		DataType: "float32",
		Shape:    classProbsShape,
		Extra:    extra,
	}
	out3 := mlmodel.TensorInfo{
		Name:     "score",
		DataType: "float32",
		Shape:    scoresShape,
		Extra:    extra,
	}
	outputs = append(outputs, out1, out2, out3)
	md.Outputs = outputs
	return md
}

// func convertInt64SliceToInt(sliceInt64 []int64) []int {
// 	sliceInt := make([]int, 0, len(sliceInt64))
// 	for _, v := range sliceInt64 {
// 		sliceInt = append(sliceInt, int(v))
// 	}
// 	return sliceInt
// }
