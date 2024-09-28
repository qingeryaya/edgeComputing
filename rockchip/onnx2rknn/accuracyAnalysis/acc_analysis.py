from dumpOnnxLayerOutput import ONNXInference
from dumpOnnxLayerOutput import modify_model_to_output_intermediate_layers
from dumpRKNNOutput import RKNNInfer

onnx_model_path = "yolov5s.onnx"
onnx_model_path_mod = onnx_model_path.replace(".onnx","_mod.onnx")
input_size=(640,640)
src_acc_analysis = "000000000241.jpg"

modify_model_to_output_intermediate_layers(onnx_model_path,onnx_model_path_mod)
onnxInfer_ori = ONNXInference(onnx_model_path, input_size)
onnxInfer_mod = ONNXInference(onnx_model_path_mod, input_size)
onnxInfer_ori.dump_output_to_txt(src_acc_analysis)
onnxInfer_mod.dump_all_layer_outputs_to_txt(src_acc_analysis)
rknnInfer= RKNNInfer(
    onnx_model_path=onnx_model_path,
    input_size=input_size,
    is_save_rknn=True,
    do_quantization=True,
    src_acc_analysis=src_acc_analysis)
rknnInfer.buildRKNNAndAccAnalysis()