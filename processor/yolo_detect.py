from ultralytics import YOLO
import cv2

video_path = f"/home/tdt/Documents/project/HumanPoseClassify/assets/videos/zz/video1.mp4"

model = YOLO('yolo11m-pose')

# 使用cv2读取视频
cap = cv2.VideoCapture(video_path)
# 读取视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 使用模型检测z
    results = model(frame)
        # 绘制结果
    for result in results:
        # 检查是否存在关键点输出
        if result.keypoints is not None:
            # 将关键点转换为 NumPy 数组
            keypoints = result.keypoints.cpu().numpy()  # 假设你在使用 CUDA
            
            # 打印 keypoints 的形状和内容
            print("关键点形状:", keypoints.shape)  # 打印 keypoints 的形状
            print("关键点内容:", keypoints)  # 打印 keypoints 内容
            
            for person_keypoints in keypoints:  # 对每个人的关键点进行处理
                for i, keypoint in enumerate(person_keypoints):					
                    print(f"关键点原始数据 {i}: {keypoint}")  # 打印每个关键点的原始数据
                    if len(keypoint) >= 2:  # 检查 keypoint 是否有至少两个值
                        x, y = keypoint[:2]  # 只取前两个值 x 和 y
                        x, y = int(x * frame.shape[1]), int(y * frame.shape[0])  # 归一化到图像坐标
                        print(f"关键点 {i}: x = {x:.2f}, y = {y:.2f}")
                        # 在图像上绘制关键点
                        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # 绘制圆点，标记关键点
                    else:
                        print(f"警告: 关键点 {i} 只包含 {len(keypoint)} 个值，无法解包")
    # 显示结果
    cv2.imshow('YOLOv8 Pose Detection', frame)
    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
