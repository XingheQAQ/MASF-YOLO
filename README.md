# MASF-YOLO: An Improved YOLOv11 Network for Small Object Detection on Drone View
MASF-YOLO is an enhanced object detection network based on YOLOv11, specifically designed for small object detection in UAV imagery.arXiv:https://arxiv.org/abs/2504.18136

<div align="center">
  <img width="770" height="760" alt="MASF-YOLO Architecture" src="https://github.com/user-attachments/assets/512c6f1d-9917-45a8-8ff7-2c42a8cc58e6" />
</div>

# Contributions
Multi-scale Feature Aggregation Module (MFAM)
<div align="center">
<img width="664" height="471" alt="image" src="https://github.com/user-attachments/assets/cd736e9d-cde7-4ba7-ae54-d8520eb3bfa2" />
</div>

Improved Efficient Multi-scale Attention(IEMA)
<div align="center">
<img width="1487" height="1282" alt="图片1" src="https://github.com/user-attachments/assets/03ed409b-c769-460a-8b4a-ff01bd874c08" />
</div>

Dimension-Aware Selective Integration Module(DASI)
<div align="center">
<img width="1504" height="589" alt="图片2" src="https://github.com/user-attachments/assets/b94a611a-4a52-4817-8c5c-7f491a8d0d24" />
</div>

# Dataset
https://github.com/VisDrone/VisDrone-Dataset

# Environment
CUDA Version: 11.3
GPU: NVIDIA GeForce RTX 4090D 24G

# Training configuration
Optimizer: SGD
Batch Size: 12
Total Epochs: 100
Image Size: 640×640
For complete training configuration and implementation details, please refer to the train.py

# Ablation Study
<div align="center">
<img width="1352" height="325" alt="image" src="https://github.com/user-attachments/assets/a792c777-18be-43b5-b3c0-06db309d4753" />
</div>

<div align="center">
<img width="1351" height="330" alt="image" src="https://github.com/user-attachments/assets/3c70a6e6-069e-491e-976b-4037c2ed5078" />
</div>

# Comparison With State-of-the-Arts
<div align="center">
<img width="646" height="417" alt="image" src="https://github.com/user-attachments/assets/e57946d5-dc5d-4505-a9be-0ad58a8815d7" />
</div>
