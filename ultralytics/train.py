import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
   # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(r'ultralytics/new_models/yolov11s.yaml')
    model.train(
                data=r'../ultralytics/datasets/VisDrone2019.yaml',
                imgsz=640,
                epochs=100,
                batch=12,
                close_mosaic=10,      # 表示在10个epoch之后关闭数据增强
                workers=8,
                device='0',
                resume=True,
                #优化器
                optimizer='SGD',
                momentum=0.937,
                #学习率
                cos_lr=True,
                lr0=0.01,
                lrf=0.01,
                amp=True,
                project='runs_train',
                name='yolov11s',
                single_cls=False,  # 是否是单类别检测
                cache=False,
                patience=100, # (int) epochs to wait for no observable improvement for early stopping of training
                conf=0.25,# (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
                iou=0.7, # (float) intersection over union (IoU) threshold for NMS
                )

    #model.info(detailed=True)
    #try:
        #model.profile(imgsz=[640, 640])
    #except Exception as e:
        #print(e)
        #pass
    #model.fuse()
    
    
    
    
