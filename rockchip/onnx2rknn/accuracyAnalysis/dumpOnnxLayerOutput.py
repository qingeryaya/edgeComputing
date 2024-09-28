import onnxruntime as ort
import numpy as np
import cv2
import onnx
import os


class ONNXInference:
    def __init__(self, model_path, input_shape):
        """
        初始化ONNX推理类，加载模型，设置输入形状。
        :param model_path: ONNX模型路径
        :param input_shape: 输入图像的形状 (height, width)
        """
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_shape = input_shape
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def preprocess(self, image_path):
        """
        图像预处理，适配ONNX模型输入。
        :param image_path: 输入图像路径
        :return: 预处理后的图像
        """
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.input_shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = np.expand_dims(image, axis=0)
        image /= 255.0
        return image

    def dump_output_to_txt(self, image_path):
        """
        对ONNX模型进行推理，并将输出层的结果写入txt文件。
        文件名为输出层的名字。
        :param image_path: 输入图像路径
        """
        input_data = self.preprocess(image_path)
        outputs = self.session.run(self.output_names, {self.input_name: input_data})

        # 将每个输出层的结果保存为txt文件
        for i, output in enumerate(outputs):
            output_layer_name = self.output_names[i]
            np.savetxt(f'./dump/outputLayers_onnx/{output_layer_name}.txt', output.flatten(), fmt='%.8f')
            print(f'Saved output to ./dump/outputLayers_onnx/{output_layer_name}.txt')


    def dump_all_layer_outputs_to_txt(self, image_path):
        """
        获取ONNX模型的每一层输出的结果。由于onnxruntime不支持直接获取所有中间层的输出，
        此方法仅用于获取模型的输出层结果。
        :param image_path: 输入图像路径
        """
        input_data = self.preprocess(image_path)
        outputs = self.session.run(self.output_names, {self.input_name: input_data})

        # 输出层的结果
        for i, output in enumerate(outputs):
            layer_name = self.output_names[i]
            if "/" in layer_name:
                layer_name = layer_name.replace("/","_").replace(".","_")
            try:
                np.savetxt(f'./dump/allLayers_onnx/{layer_name}.txt', output.flatten(), fmt='%.8f')
                print(f'Saved layer name:[ {layer_name} ] output to {layer_name}.txt')
            except Exception as e:
                print(f"{layer_name} layer can not dump")
        # os.remove(self.model_path)


def modify_model_to_output_intermediate_layers(onnx_model_path, output_model_path):
    model = onnx.load(onnx_model_path)
    graph = model.graph

    # 通过遍历节点添加中间层输出
    for node in graph.node:
        for output in node.output:
            graph.output.append(onnx.ValueInfoProto(name=output))

    onnx.save(model, output_model_path)
    print(f"Modified model saved to {output_model_path}")