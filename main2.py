import os
import cv2
import torch
import onnxruntime
import numpy as np
from utils import poly_postprocess, vis,min_rect,ValTransform,demo_postprocess_armor,demo_postprocess_buff
from utils import COCO_CLASSES
import time
from collections import deque

class ArmorZoomTracker:
    def __init__(self, frame_shape):
        # 初始化跟踪器参数
        self.frame_center = (frame_shape[1] // 2, frame_shape[0] // 2)  # 图像中心坐标(x, y)
        self.frame_size = frame_shape[:2]  # 原始图像尺寸(height, width)

        # 可调节的缩放参数
        self.max_zoom = 2.0          # 最大缩放倍数
        self.scale_factor = 0.45     # 面积比例因子
        self.ratio_exponent = 0.3    # 面积指数调整
        self.zoom_speed = 0.1        # 缩放平滑速度
        self.history_length = 5      # 位置历史长度

        # 跟踪状态变量
        self.current_zoom = 1.0       # 当前缩放倍数
        self.target_zoom = 1.0        # 目标缩放倍数
        self.position_history = deque(maxlen=self.history_length)  # 位置历史队列
        self.lost_counter = 0         # 目标丢失计数器
        self.max_lost_frames = 15     # 最大允许丢失帧数

    def update_armor(self, armor_rect):
        """更新装甲板位置并计算目标缩放倍数"""
        x, y, w, h = armor_rect
        # 将装甲板中心加入历史记录
        self.position_history.append((x + w // 2, y + h // 2))

        # 根据装甲板面积计算缩放倍数
        armor_area = w * h
        frame_area = self.frame_size[0] * self.frame_size[1]
        area_ratio = armor_area / frame_area

        # 调整后的面积比例和缩放倍数计算
        adjusted_ratio = area_ratio ** self.ratio_exponent
        self.target_zoom = np.clip(1.0 / (adjusted_ratio * self.scale_factor),
                                   1.0, self.max_zoom)
        self.lost_counter = 0  # 重置丢失计数器

    def smooth_zoom(self):
        """平滑过渡当前缩放倍数"""
        self.current_zoom += (self.target_zoom - self.current_zoom) * self.zoom_speed
        return np.clip(self.current_zoom, 1.0, self.max_zoom)

    def get_zoomed_frame(self, frame):
        """获取缩放后的画面及裁剪信息"""
        if self.current_zoom == 1.0:
            return frame, (0, 0, *self.frame_size[::-1]), 1.0

        # 计算裁剪区域尺寸
        zoom = self.current_zoom
        new_w = int(self.frame_size[1] / zoom)  # 裁剪区域宽度
        new_h = int(self.frame_size[0] / zoom)  # 裁剪区域高度

        # 计算平均跟踪位置（使用历史位置平滑）
        avg_center = np.mean(self.position_history, axis=0) if self.position_history else self.frame_center

        # 计算裁剪区域坐标（确保不越界）
        x = int(np.clip(avg_center[0] - new_w // 2, 0, self.frame_size[1] - new_w))
        y = int(np.clip(avg_center[1] - new_h // 2, 0, self.frame_size[0] - new_h))

        # 执行裁剪和缩放
        cropped = frame[y:y + new_h, x:x + new_w]
        zoomed_frame = cv2.resize(cropped, (self.frame_size[1], self.frame_size[0]))  # 保持原始分辨率
        return zoomed_frame, (x, y, new_w, new_h), zoom


class Predictor(object):
    def __init__(
        self,
        onnx_model_path,
        num_apex,
        num_class,
        num_color,
        cls_names=COCO_CLASSES,
        device="cpu",
        legacy=False,

    ):
        self.session = onnxruntime.InferenceSession(onnx_model_path)
        self.cls_names = cls_names
        self.num_apexes = num_apex
        self.num_classes = num_class
        self.num_colors = num_color
        self.confthre = 0.25  #conf
        self.nmsthre = 0.3    #nms
        self.test_size = (416,416)
        self.device = device
        self.preproc = ValTransform(legacy=legacy)
        self.tracker = None  # 缩放跟踪器


    def inference(self, img):
        img_info = {}

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        # 测算推理时间
        t0 = time.time()

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float().numpy()

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        feed_dict = {input_name: img}


        outputs = self.session.run([output_name], input_feed=feed_dict)

        if self.num_apexes == 4:
            outputs = demo_postprocess_armor(outputs[0], self.test_size, p6=False)[0]
        elif self.num_apexes == 5:
            outputs = demo_postprocess_buff(outputs[0], self.test_size, p6=False)[0]

        #print(outputs.dtype)
        #outputs = torch.from_numpy(outputs[0])

        bbox_preds = []
        # Convert[reg,conf,color,classes] into [bbox,conf,color and classes]
        for i in range(outputs.shape[0]):
            bbox = min_rect(outputs[i, :, :self.num_apexes * 2])
            bbox_preds.append(bbox)

        bbox_preds = torch.stack(bbox_preds)

        conf_preds = outputs[:, :, self.num_apexes * 2].unsqueeze(-1)

        cls_preds = outputs[:, :, self.num_apexes * 2 + 1 + self.num_colors:].repeat(1, 1, self.num_colors)
        # Initialize colors_preds
        colors_preds = torch.clone(cls_preds)

        for i in range(self.num_colors):
            colors_preds[:, :, i * self.num_classes:(i + 1) * self.num_classes] = outputs[:, :,
                                                                                  self.num_apexes * 2 + 1 + i:self.num_apexes * 2 + 1 + i + 1].repeat(
                1, 1, self.num_classes)

        cls_preds_converted = (colors_preds + cls_preds) / 2.0

        outputs_rect = torch.cat((bbox_preds, conf_preds, cls_preds_converted), dim=2)
        outputs_poly = torch.cat((outputs[:, :, :self.num_apexes * 2], conf_preds, cls_preds_converted), dim=2)
        # Out Format: (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        outputs = poly_postprocess(
            outputs_rect,
            outputs_poly,
            self.num_apexes,
            self.num_classes * self.num_colors,
            self.confthre,
            self.nmsthre
        )
        # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        #print("Infer time: {:.4f}ms".format((time.time() - t0) * 1000))
        return outputs, img_info


    def visual(self, output, img_info, conf=0.35):

            detections = []
            ratio = img_info["ratio"]
            img = img_info["raw_img"]
            class_names = self.cls_names
            # 初始化跟踪器
            if self.tracker is None:
                self.tracker = ArmorZoomTracker(img.shape)

            if output is None:
                return detections

            output = output.cpu()

            boxes = output[:, 0:self.num_apexes*2]
            # preprocessing: resize
            boxes /= ratio

            cls_ids = output[:, self.num_apexes*2 + 2]
            scores = output[:, self.num_apexes*2] * output[:, self.num_apexes*2 + 1]


            for i in range(len(boxes)):
                d = boxes[i]
                if self.num_apexes == 4:
                    pt0 = (int(d[0]), int(d[1]))
                    pt1 = (int(d[2]), int(d[3]))
                    pt2 = (int(d[4]), int(d[5]))
                    pt3 = (int(d[6]), int(d[7]))
                    position = (pt0, pt1, pt2, pt3)

                if self.num_apexes == 5:
                    pt0 = (int(d[0]), int(d[1]))
                    pt1 = (int(d[2]), int(d[3]))
                    pt2 = (int(d[4]), int(d[5]))
                    pt3 = (int(d[6]), int(d[7]))
                    pt4 = (int(d[8]), int(d[9]))
                    position = (pt0, pt1, pt2, pt3,pt4)

                cls_id = int(cls_ids[i])
                cls = class_names[cls_id]
                score = scores[i]
                score = round(score.item(), 2)
                if score < conf:
                    continue
                    # 按字典形式存入列表，方便调用
                detections.append({'cls': cls, 'conf': score, 'position': position})

            return detections

    def detect(self,frame):

        outputs, img_info = self.inference(frame)
        detections = self.visual(outputs[0], img_info, predictor.confthre)
        # 初始化跟踪器
        if self.tracker is None:
            self.tracker = ArmorZoomTracker(frame.shape)

        # 更新跟踪器状态
        main_rect = None
        if detections:
            # 选择面积最大的检测作为跟踪目标
            main_det = max(detections, key=lambda d:
            (d['position'][2][0] - d['position'][0][0]) *
            (d['position'][2][1] - d['position'][0][1]))
            pts = np.array(main_det['position'])
            x, y, w, h = cv2.boundingRect(pts)
            main_rect = (x, y, w, h)
            self.tracker.update_armor(main_rect)
        else:
            self.tracker.lost_counter += 1
            # 超过最大丢失帧数时重置缩放
            if self.tracker.lost_counter > self.tracker.max_lost_frames:
                self.tracker.target_zoom = 1.0

        # 应用平滑缩放
        self.tracker.smooth_zoom()
        # 获取缩放后的画面和裁剪信息
        zoomed_frame, crop_info, _ = self.tracker.get_zoomed_frame(frame)

        # 在原始图像上绘制检测框
        original_frame = self.draw_detections(frame.copy(), detections, (0, 255, 0))

        # 转换检测框坐标到缩放后的坐标系
        zoomed_detections = []
        if crop_info[2] > 0 and crop_info[3] > 0:  # 确保有效裁剪区域
            frame_height, frame_width = frame.shape[:2]
            crop_x, crop_y, new_w, new_h = crop_info
            # 计算缩放比例
            scale_x = frame_width / new_w
            scale_y = frame_height / new_h

            for d in detections:
                new_pts = []
                for (x, y) in d['position']:
                    # 坐标转换公式：
                    # 缩放后坐标 = (原始坐标 - 裁剪起点) * 缩放比例
                    dx = (x - crop_x) * scale_x
                    dy = (y - crop_y) * scale_y
                    new_pts.append((int(dx), int(dy)))
                zoomed_detections.append({
                    'cls': d['cls'],
                    'conf': d['conf'],
                    'position': new_pts
                })

        # 在缩放图像上绘制转换后的检测框
        zoomed_frame = self.draw_detections(zoomed_frame, zoomed_detections, (0, 0, 255))

        # 添加调试信息
        cv2.putText(original_frame, f"Detections: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(zoomed_frame, f"Zoom: {self.tracker.current_zoom:.1f}x", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return original_frame, zoomed_frame


    def draw_detections(self, frame, detections, color=(0, 255, 0)):
        """在图像上绘制检测框"""
        for d in detections:
            pts = np.array(d['position'], dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 2)  # 绘制四边形
            # 显示类别和置信度
            cv2.putText(frame, f"{d['cls']} {d['conf']}",
                        pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# if __name__ == "__main__":
#
#     video_path = "./video/15.mp4"
#     #video_path = "./前哨站/蓝方前哨站狙击点视角全速.mp4"
#
#     #onnx_model_path = "./model/500.onnx"
#     onnx_model_path = "./model/TUP/best_06_02.onnx"
#     #onnx_model_path = "./model/armor1000.onnx"
#     #onnx_model_path = "./model/train_1000.onnx"
#     #根据自己模型的不同可对关键点数量，颜色数量，类别数量进行相对应的修改
#     predictor = Predictor(onnx_model_path, num_apex = 4, num_class = 8,num_color = 8)
#
#     cap = cv2.VideoCapture(video_path)
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         outputs, img_info = predictor.inference(frame)
#         result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
#
#         cv2.imshow("Video", result_image)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 键退出循环
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "./video/4.avi"
    onnx_model_path = "./model/buff_300.onnx"
    predictor = Predictor(onnx_model_path, num_apex=5, num_class=2, num_color=2)

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("无法打开或视频已全部输出")
            break

        original, zoomed = predictor.detect(frame)

        # 显示结果
        cv2.imshow("Original Detection", original)
        cv2.imshow("Zoomed Detection", zoomed)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            break

    cap.release()
    cv2.destroyAllWindows()



