import torch
import torchvision
from torchvision import transforms
from detection.faster_rcnn import FastRCNNPredictor, FasterRCNN, fasterrcnn_resnet50_fpn

import onnxruntime
import onnx
import onnxruntime
from onnxruntime.quantization import quantize, QuantizationMode

device = "cpu"

####################
### Define model ###
####################
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

num_classes = 4
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

model_path = "model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))


####################
### export model ###
####################
model.eval()
dummy_input = torch.randn((1, 3, 1024, 1024), device=device)

output = model(dummy_input)
model.eval()
dummy_input = torch.randn((1, 3, 1024, 1024), device=device)

output = model(dummy_input)
model.to(device)
torch.onnx.export(model, dummy_input, "faster_rcnn.onnx", do_constant_folding=True, output_names=["box","label","score"]
                , opset_version = 11
                , dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'}
                                 ,"box":{0: 'batch'}
                                 ,"label":{0: 'batch'} 
                                 ,"score":{0: 'batch'}})

def quantize_onnx_model(onnx_model_path, quantized_model_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx
    onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QUInt8)
    # onnx.save(quantized_onnx_model,quantized_model_path)
    print(f"quantized model saved to:{quantized_model_path}")  

quantize_onnx_model("faster_rcnn.onnx","faster_rcnn_quantize.onnx")


#######################
### test onnx model ###
#######################
sess = onnxruntime.InferenceSession("faster_rcnn_quantize.onnx", providers=["CPUExecutionProvider"])
dummy_input = torch.randn((1, 3, 1280, 1280), device=device)

out = sess.run(None, {"images": dummy_input.numpy()})
