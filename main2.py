import cv2
import torch
import numpy as np
from utils import COCO_CLASSES, poly_postprocess, min_rect, ValTransform
from utils.base_infer import BaseInfer


class ArmorZoomTracker:
    """自适应缩放跟踪器，用于放大跟踪检测到的装甲板。"""

    def __init__(self, frame_shape):
        self.frame_center = (frame_shape[1] // 2, frame_shape[0] // 2)
        self.frame_size = frame_shape[:2]  # (height, width)

        # 可调节的缩放参数
        self.max_zoom = 2.0
        self.scale_factor = 0.45
        self.ratio_exponent = 0.3
        self.zoom_speed = 0.1
        self.history_length = 5

        # 跟踪状态变量
        self.current_zoom = 1.0
        self.target_zoom = 1.0
        self.position_history = []
        self.lost_counter = 0
        self.max_lost_frames = 15

    def update_armor(self, armor_rect):
        """更新装甲板位置并计算目标缩放倍数。"""
        x, y, w, h = armor_rect
        self.position_history.append((x + w // 2, y + h // 2))
        # 保持历史长度限制
        if len(self.position_history) > self.history_length:
            self.position_history = self.position_history[-self.history_length:]

        armor_area = w * h
        frame_area = self.frame_size[0] * self.frame_size[1]
        area_ratio = armor_area / frame_area

        adjusted_ratio = area_ratio ** self.ratio_exponent
        self.target_zoom = np.clip(
            1.0 / (adjusted_ratio * self.scale_factor), 1.0, self.max_zoom
        )
        self.lost_counter = 0

    def smooth_zoom(self):
        """平滑过渡当前缩放倍数。"""
        self.current_zoom += (self.target_zoom - self.current_zoom) * self.zoom_speed
        return np.clip(self.current_zoom, 1.0, self.max_zoom)

    def get_zoomed_frame(self, frame):
        """获取缩放后的画面及裁剪信息。"""
        if self.current_zoom == 1.0:
            return frame, (0, 0, *self.frame_size[::-1]), 1.0

        zoom = self.current_zoom
        new_w = int(self.frame_size[1] / zoom)
        new_h = int(self.frame_size[0] / zoom)

        avg_center = (
            np.mean(self.position_history, axis=0)
            if self.position_history
            else self.frame_center
        )

        x = int(np.clip(avg_center[0] - new_w // 2, 0, self.frame_size[1] - new_w))
        y = int(np.clip(avg_center[1] - new_h // 2, 0, self.frame_size[0] - new_h))

        cropped = frame[y:y + new_h, x:x + new_w]
        zoomed_frame = cv2.resize(cropped, (self.frame_size[1], self.frame_size[0]))
        return zoomed_frame, (x, y, new_w, new_h), zoom


class ZoomPredictor(BaseInfer):
    """带缩放跟踪功能的推理器。

    继承 BaseInfer 的推理能力，额外提供：
    - 自动选择最大装甲板作为跟踪目标
    - 平滑缩放放大跟踪区域
    - 缩放坐标系下的检测结果转换
    """

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
        super().__init__(
            onnx_model_path, num_apex, num_class, num_color, device, legacy
        )
        self.cls_names = cls_names
        self.tracker = None

    def visual(self, output, img_info, conf=0.35):
        """解析检测结果为字典列表。"""
        detections = []
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        class_names = self.cls_names

        if self.tracker is None:
            self.tracker = ArmorZoomTracker(img.shape)

        if output is None:
            return detections

        output = output.cpu()

        boxes = output[:, 0:self.num_apexes * 2]
        boxes /= ratio

        cls_ids = output[:, self.num_apexes * 2 + 2]
        scores = output[:, self.num_apexes * 2] * output[:, self.num_apexes * 2 + 1]

        for i in range(len(boxes)):
            d = boxes[i]
            # 使用元组列表格式（与 main2.py 原始逻辑一致）
            position = tuple(
                (int(d[2 * j]), int(d[2 * j + 1])) for j in range(self.num_apexes)
            )

            cls_id = int(cls_ids[i])
            cls = class_names[cls_id]
            score = round(scores[i].item(), 2)
            if score < conf:
                continue

            detections.append({"cls": cls, "conf": score, "position": position})

        return detections

    def detect(self, frame):
        """检测并应用缩放跟踪，返回原始标注帧和缩放标注帧。"""
        outputs, img_info = self.inference(frame)
        detections = self.visual(outputs[0], img_info, self.confthre)

        if self.tracker is None:
            self.tracker = ArmorZoomTracker(frame.shape)

        # 更新跟踪器状态
        if detections:
            main_det = max(
                detections,
                key=lambda d: (d["position"][2][0] - d["position"][0][0])
                * (d["position"][2][1] - d["position"][0][1]),
            )
            pts = np.array(main_det["position"])
            x, y, w, h = cv2.boundingRect(pts)
            self.tracker.update_armor((x, y, w, h))
        else:
            self.tracker.lost_counter += 1
            if self.tracker.lost_counter > self.tracker.max_lost_frames:
                self.tracker.target_zoom = 1.0

        # 应用平滑缩放
        self.tracker.smooth_zoom()
        zoomed_frame, crop_info, _ = self.tracker.get_zoomed_frame(frame)

        # 在原始图像上绘制检测框
        original_frame = self.draw_detections(frame.copy(), detections, (0, 255, 0))

        # 转换检测框坐标到缩放后的坐标系
        zoomed_detections = []
        if crop_info[2] > 0 and crop_info[3] > 0:
            frame_height, frame_width = frame.shape[:2]
            crop_x, crop_y, new_w, new_h = crop_info
            scale_x = frame_width / new_w
            scale_y = frame_height / new_h

            for d in detections:
                new_pts = []
                for (x, y) in d["position"]:
                    dx = (x - crop_x) * scale_x
                    dy = (y - crop_y) * scale_y
                    new_pts.append((int(dx), int(dy)))
                zoomed_detections.append(
                    {"cls": d["cls"], "conf": d["conf"], "position": new_pts}
                )

        # 在缩放图像上绘制转换后的检测框
        zoomed_frame = self.draw_detections(zoomed_frame, zoomed_detections, (0, 0, 255))

        # 添加调试信息
        cv2.putText(
            original_frame,
            f"Detections: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            zoomed_frame,
            f"Zoom: {self.tracker.current_zoom:.1f}x",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        return original_frame, zoomed_frame

    def draw_detections(self, frame, detections, color=(0, 255, 0)):
        """在图像上绘制检测框。"""
        for d in detections:
            pts = np.array(d["position"], dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 2)  # type: ignore[arg-type]
            cv2.putText(
                frame,
                f"{d['cls']} {d['conf']}",
                pts[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return frame


if __name__ == "__main__":
    video_path = "./video/4.avi"
    onnx_model_path = "./model/buff_300.onnx"
    predictor = ZoomPredictor(onnx_model_path, num_apex=5, num_class=2, num_color=2)

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("无法打开或视频已全部输出")
            break

        original, zoomed = predictor.detect(frame)

        cv2.imshow("Original Detection", original)
        cv2.imshow("Zoomed Detection", zoomed)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
