package yoloonnx

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"path"
	"runtime"
	"sort"
	"time"

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
	YoloOnnxCpu      = resource.NewModel("hipsterbrown", "mlmodel", "yolo-onnx")
	errUnimplemented = errors.New("unimplemented")
)

var imageHeight, imageWidth int

const DEBUG_INPUT = false
const IMAGE_SIZE = 576 // 90% of the input size which keeps the gRPC message from exceeding the 4MB limit
const INPUT_SIZE = 640

// DataTypeMap maps the long ONNX data type labels to the data type as written in Go.
var DataTypeMap = map[ort.TensorElementDataType]string{
	ort.TensorElementDataTypeFloat: "float32",
	ort.TensorElementDataTypeUint8: "uint8",
}

func init() {
	resource.RegisterService(mlmodel.API, YoloOnnxCpu,
		resource.Registration[mlmodel.Service, *Config]{
			Constructor: newYoloOnnxYoloOnnxCpu,
		},
	)
}

type Config struct {
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
	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, err
	}

	out0OrtTensorInfo := outputInfo[0]
	outputShape0 := ort.NewShape(out0OrtTensorInfo.Dimensions...)
	outputTensor0, err := ort.NewEmptyTensor[float32](outputShape0)
	if err != nil {
		return nil, err
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}

	if runtime.GOOS == "darwin" {
		err = options.AppendExecutionProviderCoreML(0)
		if err != nil {
			inputTensor.Destroy()
			outputTensor0.Destroy()
			return nil, fmt.Errorf("Error enabling CoreML: %w", err)
		}
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
		// Get the shape of the input tensor
		shape := imageTensor.Shape()
		imageHeight, imageWidth = shape[2], shape[3]
		if imageHeight != INPUT_SIZE || imageWidth != INPUT_SIZE {
			// Need to pad from image height/width to INPUT_SIZE
			paddedData := make([]float32, 1*3*INPUT_SIZE*INPUT_SIZE)

			// Calculate padding offsets to center the original image
			offsetY := (INPUT_SIZE - imageHeight) / 2
			offsetX := (INPUT_SIZE - imageWidth) / 2

			// Copy the original data to the center of the padded tensor
			for c := range 3 { // For each channel
				for y := range imageHeight {
					for x := range imageWidth {
						srcIdx := ((0*3+c)*imageHeight+y)*imageWidth + x
						dstIdx := ((0*3+c)*INPUT_SIZE+(y+offsetY))*INPUT_SIZE + (x + offsetX)
						paddedData[dstIdx] = float32Data[srcIdx]
					}
				}
			}

			if DEBUG_INPUT {
				width, height := INPUT_SIZE, INPUT_SIZE
				img := image.NewRGBA(image.Rect(0, 0, width, height))
				channelSize := width * height

				for y := range height {
					for x := range width {
						i := y*width + x
						// CHW format: C = 3 (RGB)
						r := paddedData[i] * 255
						g := paddedData[channelSize+i] * 255
						b := paddedData[2*channelSize+i] * 255

						// Convert float32 [0,1] or [0,255] to uint8 (scale if needed)
						img.Set(x, y, color.RGBA{
							R: uint8(clamp(r, 0, 255)),
							G: uint8(clamp(g, 0, 255)),
							B: uint8(clamp(b, 0, 255)),
							A: 255,
						})
					}
				}

				outFile, err := os.Create(fmt.Sprintf("padded_input_%s.jpeg", time.Now()))
				if err != nil {
					return nil, err
				}
				defer outFile.Close()

				err = jpeg.Encode(outFile, img, &jpeg.Options{Quality: 90})
				if err != nil {
					return nil, err
				}
			}

			return paddedData, nil
		}
		return float32Data, nil
	}
	return nil, errors.Errorf("input tensor must be of tensor type float32, got %v", imageTensor.Dtype())
}

type boundingBox struct {
	classId        float32
	confidence     float32
	x1, y1, x2, y2 float32
}

func (b *boundingBox) String() string {
	return fmt.Sprintf("Object %f (confidence %f): (%f, %f), (%f, %f)",
		b.classId, b.confidence, b.x1, b.y1, b.x2, b.y2)
}

func (b *boundingBox) toRect() image.Rectangle {
	return image.Rect(int(b.x1), int(b.y1), int(b.x2), int(b.y2)).Canon()
}

func (b *boundingBox) rectArea() int {
	size := b.toRect().Size()
	return size.X * size.Y
}

func (b *boundingBox) intersection(other *boundingBox) float32 {
	r1 := b.toRect()
	r2 := other.toRect()
	intersected := r1.Intersect(r2).Canon().Size()
	return float32(intersected.X * intersected.Y)
}

func (b *boundingBox) union(other *boundingBox) float32 {
	intersectArea := b.intersection(other)
	totalArea := float32(b.rectArea() + other.rectArea())
	return totalArea - intersectArea
}

func (b *boundingBox) iou(other *boundingBox) float32 {
	return b.intersection(other) / b.union(other)
}

func (yolo *yoloOnnxYoloOnnxCpu) processOutput(output []float32) (ml.Tensors, error) {
	// there is 1 output tensor. Turn it into 3 tensors with the right backing
	outputShape := yolo.session.Output.GetShape()
	if len(outputShape) != 3 {
		return nil, fmt.Errorf("unexpected output shape for YOLO model: %v", outputShape)
	}
	numChannels := int(outputShape[1])
	numBoxes := int(outputShape[2])
	numClasses := numChannels - 4

	boxesData := make([]float32, numBoxes*4)
	scoresData := make([]float32, numBoxes)
	classProbsData := make([]float32, numBoxes)

	var classId float32
	var probability float32

	boundingBoxes := make([]boundingBox, 0, numBoxes)

	offsetX := (INPUT_SIZE - imageWidth) / 2
	offsetY := (INPUT_SIZE - imageHeight) / 2
	scaleX := float32(INPUT_SIZE) / float32(imageWidth)
	scaleY := float32(INPUT_SIZE) / float32(imageHeight)

	for idx := range numBoxes {
		probability = float32(-1e9)

		for col := range numClasses {
			currentProb := output[numBoxes*(col+4)+idx]
			if currentProb > probability {
				probability = currentProb
				classId = float32(col)
			}
		}

		xc, yc := output[idx], output[numBoxes+idx]
		w, h := output[2*numBoxes+idx], output[3*numBoxes+idx]
		x1 := (xc - w/2)
		y1 := (yc - h/2)
		x2 := (xc + w/2)
		y2 := (yc + h/2)

		adjustedX1 := (x1 - float32(offsetX)) * scaleX
		adjustedY1 := (y1 - float32(offsetY)) * scaleY
		adjustedX2 := (x2 - float32(offsetX)) * scaleX
		adjustedY2 := (y2 - float32(offsetY)) * scaleY

		boundingBoxes = append(boundingBoxes, boundingBox{
			classId:    classId,
			confidence: probability,
			x1:         adjustedX1,
			y1:         adjustedY1 * 0.75,
			x2:         adjustedX2,
			y2:         adjustedY2 * 0.75,
		})
	}

	sort.Slice(boundingBoxes, func(i, j int) bool {
		return boundingBoxes[i].confidence > boundingBoxes[j].confidence
	})

	mergedResults := make([]boundingBox, 0, len(boundingBoxes))

	for _, candidateBox := range boundingBoxes {
		overlapsExistingBox := false
		for _, existingBox := range mergedResults {
			if (&candidateBox).iou(&existingBox) > 0.7 {
				overlapsExistingBox = true
				break
			}
		}
		if !overlapsExistingBox {
			mergedResults = append(mergedResults, candidateBox)
		}
	}

	for i, bbox := range mergedResults {
		classProbsData[i] = bbox.classId
		scoresData[i] = bbox.confidence
		boxesData[i*4] = bbox.y1
		boxesData[i*4+1] = bbox.x1
		boxesData[i*4+2] = bbox.y2
		boxesData[i*4+3] = bbox.x2
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
			return "./third_party/onnxruntime.dll", nil
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.dylib", nil
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.so", nil
		}
		return "./third_party/onnxruntime.so", nil
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
		Shape:    []int{1, 3, IMAGE_SIZE, IMAGE_SIZE},
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

func clamp(v float32, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
