import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'../MASF-YOLO/best_s.pt')  # 导入模型权重
    model.val(
                data=r'../ultralytics/datasets/VisDrone2019.yaml',
                imgsz=640,
                batch=20,
                split='val',
                workers=8,
                device='0',
                project='runs_val',
                name='yolov11s',
                max_det=300,
                conf=0.25,# (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
                iou=0.7, # (float) intersection over union (IoU) threshold for NMS
                )
    
