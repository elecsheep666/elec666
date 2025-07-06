################################################
#  test one Image
################################################

import cv2
from ultralytics import YOLO
import supervision as sv

image_path = f"xianyu.png"
image = cv2.imread(image_path)

model = YOLO('best.pt')

# results = model(image, verbose=False)[0]
# 在模型预测阶段调整参数
results = model(image,
                iou=0.45,       # 调高NMS的IOU阈值，减少重叠框
                device='cpu',
                verbose=False)[0]
detections = sv.Detections.from_ultralytics(results)
# 创建标注器
# 检测框
box_annotator = sv.BoxAnnotator()
# 标签和置信度
label_annotator = sv.LabelAnnotator()

# 创建原始图像的副本，避免修改原始图像
annotated_image = image.copy()
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# 使用 supervision 的绘图功能显示标注后的图像
sv.plot_image(annotated_image)
