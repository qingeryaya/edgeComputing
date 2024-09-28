from rknn.api import RKNN


class RKNNInfer:
    def __init__(self, onnx_model_path,input_size, is_save_rknn,do_quantization,src_acc_analysis):
        self.onnx_model_path = onnx_model_path
        self.input_size = input_size
        self.is_save_rknn = is_save_rknn
        self.do_quantization = do_quantization
        self.src_acc_analysis = src_acc_analysis
        self.rknn = RKNN(verbose=True, verbose_file='logs/yolov5net_build.log')
        self.rknn.config(
                    mean_values=[[0., 0., 0.]],
                    std_values=[[255., 255., 255.]],
                    quant_img_RGB2BGR=False,
                    quantized_dtype='w8a8',  # 权重为8bit非对称量化精度，激活值为8bit非对称量化精度
                    quantized_algorithm="mmse",  # 3588 支持三种量化算法 normal KL散度，mmse
                    quantized_method="channel",
                    float_dtype='float16',
                    optimization_level=3,
                    target_platform='rk3588',
                    custom_string="rknn yolov5 test",
                    single_core_mode=True
                )
        
    def buildRKNNAndAccAnalysis(self):
        ret = self.rknn.load_onnx(
                model=self.onnx_model_path,
                inputs=["images"],
                input_size_list=[[1,3,self.input_size[0],self.input_size[1]]],
                outputs=["output0","343","345"]
            )
        ret = self.rknn.build(
            do_quantization=self.do_quantization,
            dataset='./dataset.txt'
        )
        self.rknn.accuracy_analysis(
            [self.src_acc_analysis,],
            "./dump/dump_rknn",
            target="rk3588",
        )
    
        

    