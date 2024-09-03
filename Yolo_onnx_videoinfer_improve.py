import os
import cv2
import numpy as np
import onnxruntime
import time
import math

CLASSES = ['person']  # 定义类别名称，仅包括 'person' 类

class YOLOV5():
    def __init__(self, onnxpath):
        # 初始化 YOLOV5 模型
        self.onnx_session = onnxruntime.InferenceSession(onnxpath)  # 创建 ONNX 运行环境
        self.input_name = self.get_input_name()  # 获取输入名称
        self.output_name = self.get_output_name()  # 获取输出名称

    def get_input_name(self):
        # 获取模型的输入名称
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)  # 收集所有输入的名称
        return input_name

    def get_output_name(self):
        # 获取模型的输出名称
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)  # 收集所有输出的名称
        return output_name

    def get_input_feed(self, img_tensor):
        # 为每个输入名称分配图像数据
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = img_tensor  # 为每个输入名称分配图像数据
        return input_feed

    def inference(self, img):
        # 执行模型推理
        or_img = cv2.resize(img, (640, 640))  # 调整图像大小到640x640
        img = or_img[:, :, ::-1].transpose(2, 0, 1)  # 将图像从 BGR 转换为 RGB 并调整维度（HWC 到 CHW）
        img = img.astype(dtype=np.float32)  # 转换数据类型为 float32
        img /= 255.0  # 归一化图像数据到 [0, 1]
        img = np.expand_dims(img, axis=0)  # 增加批次维度
        input_feed = self.get_input_feed(img)  # 获取输入数据
        pred = self.onnx_session.run(None, input_feed)[0]  # 执行推理，获取模型预测
        return pred, or_img

def nms(dets, thresh):
    # 非极大抑制 (NMS) 以去除重叠的边界框
    x1 = dets[:, 0]  # 目标框的左上角 x 坐标
    y1 = dets[:, 1]  # 目标框的左上角 y 坐标
    x2 = dets[:, 2]  # 目标框的右下角 x 坐标
    y2 = dets[:, 3]  # 目标框的右下角 y 坐标
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)  # 计算框的面积
    scores = dets[:, 4]  # 置信度分数
    keep = []  # 存储保留的框的索引
    index = scores.argsort()[::-1]  # 根据分数对框进行排序

    while index.size > 0:
        i = index[0]  # 选择分数最高的框
        keep.append(i)  # 保留该框
        x11 = np.maximum(x1[i], x1[index[1:]])  # 计算相交区域的左上角 x 坐标
        y11 = np.maximum(y1[i], y1[index[1:]])  # 计算相交区域的左上角 y 坐标
        x22 = np.minimum(x2[i], x2[index[1:]])  # 计算相交区域的右下角 x 坐标
        y22 = np.minimum(y2[i], y2[index[1:]])  # 计算相交区域的右下角 y 坐标

        w = np.maximum(0, x22 - x11 + 1)  # 计算相交区域的宽度
        h = np.maximum(0, y22 - y11 + 1)  # 计算相交区域的高度

        overlaps = w * h  # 计算相交区域的面积
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)  # 计算 IOU
        idx = np.where(ious <= thresh)[0]  # 保留 IOU 小于阈值的框
        index = index[idx + 1]  # 继续处理剩下的框
    return keep  # 返回保留的框的索引

def xywh2xyxy(x):
    # 将 [x, y, w, h] 格式转换为 [x1, y1, x2, y2] 格式
    y = np.copy(x)  # 复制输入数组
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # 计算左上角 x 坐标
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # 计算左上角 y 坐标
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # 计算右下角 x 坐标
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # 计算右下角 y 坐标
    return y

# def filter_box(org_box, conf_thres, iou_thres):
#     # 过滤边界框，执行 NMS 和置信度筛选
#     org_box = np.squeeze(org_box)  # 删除所有单维度
#     conf = org_box[..., 4] > conf_thres  # 根据置信度阈值进行筛选
#     box = org_box[conf == True]  # 选择置信度大于阈值的框
#
#     cls_cinf = box[..., 5:]  # 获取类别得分
#     cls = []
#     for i in range(len(cls_cinf)):
#         cls.append(int(np.argmax(cls_cinf[i])))  # 选择置信度最高的类别
#     all_cls = list(set(cls))  # 获取所有唯一类别
#
#     output = []
#     for i in range(len(all_cls)):
#         curr_cls = all_cls[i]  # 当前类别
#         curr_cls_box = []
#         curr_out_box = []
#         for j in range(len(cls)):
#             if cls[j] == curr_cls:
#                 box[j][5] = curr_cls  # 替换类别下标
#                 curr_cls_box.append(box[j][:6])  # 添加当前类别的框
#         curr_cls_box = np.array(curr_cls_box)  # 转换为数组
#         curr_cls_box = xywh2xyxy(curr_cls_box)  # 转换坐标格式
#         curr_out_box = nms(curr_cls_box, iou_thres)  # 执行非极大抑制
#         for k in curr_out_box:
#             output.append(curr_cls_box[k])  # 添加保留的框
#     output = np.array(output)  # 转换为数组
#     return output
def filter_box(org_box, conf_thres, iou_thres):
    org_box = np.squeeze(org_box)  # 删除所有单维度
    if len(org_box.shape) < 2:
        print("Error: org_box does not have the expected shape.")
        return np.array([])

    conf = org_box[..., 4] > conf_thres  # 根据置信度阈值进行筛选
    box = org_box[conf == True]  # 选择置信度大于阈值的框

    if len(box) == 0:
        return np.array([])

    cls_cinf = box[..., 5:]  # 获取类别得分
    cls = [int(np.argmax(cls_cinf[i])) for i in range(len(cls_cinf))]
    all_cls = list(set(cls))  # 获取所有唯一类别

    output = []
    for i in range(len(all_cls)):
        curr_cls = all_cls[i]  # 当前类别
        curr_cls_box = [box[j][:6] for j in range(len(cls)) if cls[j] == curr_cls]
        curr_cls_box = np.array(curr_cls_box)  # 转换为数组
        curr_cls_box = xywh2xyxy(curr_cls_box)  # 转换坐标格式
        curr_out_box = nms(curr_cls_box, iou_thres)  # 执行非极大抑制
        output.extend(curr_cls_box[k] for k in curr_out_box)  # 添加保留的框

    output = np.array(output)  # 转换为数组
    # [x_min, y_min, x_max, y_max, conf, cls]这是 output 包含的信息
    return output


# def draw(image, box_data, line_start, line_end, count):
#     # 在图像上绘制边界框和计数
#     boxes = box_data[..., :4].astype(np.int32)  # 获取框的坐标并转为整型
#     scores = box_data[..., 4]  # 获取置信度
#     classes = box_data[..., 5].astype(np.int32)  # 获取类别下标
#
#     for box, score, cl in zip(boxes, scores, classes):
#         top, left, right, bottom = box  # 解包框坐标
#         print('class: {}, score: {}'.format(CLASSES[cl], score))  # 打印类别和分数
#         print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))  # 打印框坐标
#
#         cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)  # 绘制矩形框
#         cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
#                     (top, left),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6, (0, 0, 255), 2)  # 添加文本标签
#
#     # 绘制线
#     cv2.line(image, line_start, line_end, (0, 255, 0), 5)
#     cv2.putText(image, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # 显示计数

def get_centroid(box):
    # 计算边界框的中心点
    # 这里的xy是左上角和右下角的坐标
    x1, y1, x2, y2 = box[:4]  # 确保只解包前四个值
    centroid_x = (x1 + x2) / 2
    centroid_y = (y1 + y2) / 2
    return (centroid_x, centroid_y)

def intersect_line_segment(p1, p2, q1, q2):
    # 计算线段 (p1, p2) 和 (q1, q2) 是否相交
    def orientation(p, q, r):
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    def on_segment(p, q, r):
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, q1, p2): return True
    if o2 == 0 and on_segment(p1, q2, p2): return True
    if o3 == 0 and on_segment(q1, p1, q2): return True
    if o4 == 0 and on_segment(q1, p2, q2): return True

    return False

def calculate_distance(point1, point2):
    # 计算两个点之间的欧几里得距离
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def point_to_line_distance(point, line_start, line_end):
    # 计算点到线段的垂直距离
    px, py = point
    sx, sy = line_start
    ex, ey = line_end

    line_magnitude = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
    if line_magnitude == 0:
        return float('inf')

    u = ((px - sx) * (ex - sx) + (py - sy) * (ey - sy)) / float(line_magnitude ** 2)
    if u < 0 or u > 1:
        # 如果 u < 0 或 u > 1, 点在线段的延长线上，不在实际的线段上
        return float('inf')

    ix = sx + u * (ex - sx)
    iy = sy + u * (ey - sy)
    distance = math.sqrt((ix - px) ** 2 + (iy - py) ** 2)
    return distance


def draw(image, box_data, line_start, line_end, count):
    print(f"box_data shape: {box_data.shape}")  # 打印形状用于调试
    if len(box_data.shape) < 2 or box_data.shape[1] < 6:
        print("Error: box_data does not have the expected shape or number of columns.")
        return

    boxes = box_data[..., :4].astype(np.int32)  # 获取框的坐标并转为整型
    scores = box_data[..., 4]  # 获取置信度
    classes = box_data[..., 5].astype(np.int32)  # 获取类别下标

    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box  # 解包框坐标
        print('class: {}, score: {}'.format(CLASSES[cl], score))  # 打印类别和分数
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))  # 打印框坐标

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)  # 绘制矩形框
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)  # 添加文本标签

    # 绘制线
    cv2.line(image, line_start, line_end, (0, 255, 0), 5)
    cv2.putText(image, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # 显示计数


def main():
    onnx_path = 'VOCdata/ziliao/best.onnx'  # ONNX 模型的路径
    model = YOLOV5(onnx_path)  # 创建 YOLOV5 模型实例

    video_path = 'VOCdata/kkk.mp4'
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(0)  # 使用摄像头

    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    if fps == 0:  # 如果未能获取帧率，默认设置为 30
        fps = 90
    delay = int(1000 / fps)  # 计算每帧之间的延迟

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编解码器
    out_video = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))  # 创建 VideoWriter 对象

    offset = 55  # 增加偏移量，将线向上移动更多的像素
    object_ids = {}  # 用于跟踪对象的字典
    object_counter = {}  # 用于记录对象是否已经计数
    count = 0

    start_time = time.time()  # 开始时间
    with open('count.txt', 'w') as file:
        while True:
            ret, frame = cap.read()  # 读取视频帧
            if not ret:
                break  # 如果没有读取到帧，退出循环

            frame_height, frame_width, _ = frame.shape  # 获取视频帧的宽度和高度
            line_y_position = frame_height // 2 - offset  # 将线向上移动 offset 像素

            line_start = (0, line_y_position)  # 动态设置线的起点为窗口左侧
            line_end = (frame_width, line_y_position)  # 动态设置线的终点为窗口右侧

            output, or_img = model.inference(frame)  # 执行推理

            print(f"Inference output shape: {output.shape}")  # 打印推理输出的形状
            outbox = filter_box(output, 0.5, 0.5)  # 过滤边界框

            print(f"Filtered box shape: {outbox.shape}")  # 打印过滤后的框的形状

            for box in outbox:
                centroid = get_centroid(box)  # 计算中心点

                # 寻找最近的已知对象，更新其位置
                min_distance = float('inf')
                closest_object_id = None
                for object_id, prev_centroid in object_ids.items():
                    distance = calculate_distance(centroid, prev_centroid)
                    if distance < min_distance:
                        min_distance = distance
                        closest_object_id = object_id

                if min_distance < 50:
                    object_ids[closest_object_id] = centroid
                else:
                    new_id = len(object_ids) + 1
                    object_ids[new_id] = centroid
                    closest_object_id = new_id

                # 判断对象的中心是否接近线
                if closest_object_id not in object_counter and point_to_line_distance(centroid, line_start, line_end) < 10:
                    object_counter[closest_object_id] = True
                    count += 1

            draw(or_img, outbox, line_start, line_end, count)  # 绘制结果
            out_video.write(or_img)  # 保存帧到视频文件
            cv2.imshow('Object Detection', or_img)  # 显示结果

            elapsed_time = time.time() - start_time
            print(f"Elapsed Time: {elapsed_time:.2f} seconds")

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break  # 按 'q' 键退出

        # 将总数写入 count.txt
        file.write(f"Total count: {count}\n")

    cap.release()  # 释放视频捕获对象
    out_video.release()  # 释放 VideoWriter 对象
    cv2.destroyAllWindows()  # 关闭所有窗口

if __name__ == "__main__":
    main()



