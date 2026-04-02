import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
# 模型推理
    
if __name__ == '__main__':
    model = YOLO('../MASF-YOLO/best_s.pt')
    model.predict(
                source='images',   # 图片文件
                imgsz=640,
                device='0',
                project='runs_predict',
                name='scconv',
                )

