from rknn.api import RKNN
import cv2
import time

rknn = RKNN(verbose=True, verbose_file='logs/yolov5net_build.log')
rknn.load_rknn("yolov5s.rknn")
rknn.init_runtime(
    target="rk3588",
    eval_mem=False,
)

image = cv2.imread("000000000241.jpg")
image = cv2.resize(image,(640,640))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
times = 0
while 1:
    outputs = rknn.inference(
        inputs=[image],
        data_format="nhwc"
    )
    print(f"times: {times}")
    times+=1
    
rknn.release()