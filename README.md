# Module yolo-onnx 

This [module](https://docs.viam.com/registry/modular-resources/) implements the [`rdk:service:mlmodel` API](https://docs.viam.com/appendix/apis/services/mlmodel/) for [YOLO](https://docs.ultralytics.com/models/) models exported in the [ONNX format](https://docs.ultralytics.com/modes/export/#export-formats).

With this module, you can use YOLO-based ML models from the community or ones you train yourself and integrate seamlessly with the [mlmodel vision service](https://docs.viam.com/operate/reference/services/vision/mlmodel/).

## Model hipsterbrown:mlmodel:yolo-onnx

This service includes an embedded ONNX runtime for the target OS and architecture and maps the default output from YOLO models to the expected output for the mlmodel vision service.

### Configuration
The following attribute template can be used to configure this model:

```json
{
"model_path": <string>,
"label_path": <string>
}
```

#### Attributes

The following attributes are available for this model:

| Name          | Type   | Inclusion | Description                |
|---------------|--------|-----------|----------------------------|
| `model_path` | string  | Required  | The file path to the `.onnx` model, this is filled in when selecting a model from the registry. |
| `label_path` | string | Optional  | The file path to the text file with the associated labels for class IDs used to train the model, this is filled in when selecting a model from the registry. |

#### Example Configuration

```json
{
  "model_path": "./yolov8.onnx",
  "label_path": "./coco.txt"
}
```

