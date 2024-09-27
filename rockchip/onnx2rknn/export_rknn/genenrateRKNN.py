from rknn.api import RKNN

rknn = RKNN(verbose=True, verbose_file='logs/yolov5net_build.log')
rknn.config(
    mean_values=[[0., 0., 0.]],
    std_values=[[255., 255., 255.]],
    quant_img_RGB2BGR=False,
    quantized_dtype='w8a8',  # 权重为8bit非对称量化精度，激活值为8bit非对称量化精度
    quantized_algorithm="normal",  # 3588 支持三种量化算法 normal KL散度，mmse
    quantized_method="channel",
    float_dtype='float16',
    optimization_level=3,
    target_platform='rk3588',
    custom_string="rknn yolov5 test",
    single_core_mode=True

)
ret = rknn.load_onnx(model='./yolov5s.onnx')
ret = rknn.build(do_quantization=True,
                 dataset='./dataset.txt')
ret = rknn.export_rknn(export_path='./yolov5s.rknn')
rknn.release()
