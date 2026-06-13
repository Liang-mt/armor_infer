import math

import cv2
import numpy as np
from utils.base_infer import BaseInfer
from utils.mlp_predict import NumberClassifier


class Infer2(BaseInfer):
    """两阶段推理类：YOLOX 检测 + MLP 数字识别。

    第一阶段使用 YOLOX 检测装甲板位置，
    第二阶段通过透视变换提取 ROI 并用 MLP 识别数字。
    """

    def __init__(
        self,
        onnx_model_path,
        num_apex,
        num_class,
        num_color,
        device="cpu",
        legacy=False,
        mlp_model_path="./model/mlp.onnx",
    ):
        super().__init__(onnx_model_path, num_apex, num_class, num_color, device, legacy)
        self.cls_name = ["B", "R", "O"]
        self.number_id = [
            "1", "2", "3", "4", "5", "outpost", "guard", "base", "negative",
        ]
        self.number = NumberClassifier(mlp_model_path)

    def visual(self, output, img_info, conf=0.35):
        """解析检测结果为字典列表（3类模型直接用 cls_name 索引）。"""
        detections = []
        ratio = img_info["ratio"]

        if output is None:
            return detections

        output = output.cpu()

        boxes = output[:, 0:self.num_apexes * 2]
        boxes /= ratio

        cls_ids = output[:, self.num_apexes * 2 + 2]
        scores = output[:, self.num_apexes * 2] * output[:, self.num_apexes * 2 + 1]

        for i in range(len(boxes)):
            d = boxes[i]
            position = self._parse_position(d)

            cls_id = int(cls_ids[i])
            cls = self.cls_name[cls_id]

            score = round(scores[i].item(), 2)
            if score < conf:
                continue

            detections.append({"cls": cls, "conf": score, "position": position})

        return detections

    def detect(self, frame):
        """对单帧进行检测，返回标注图像和检测结果。"""
        outputs, img_info = self.inference(frame)
        detections = self.visual(outputs[0], img_info, self.confthre)

        original_frame = self.draw_detections(frame.copy(), detections, (0, 255, 0))

        return original_frame, detections

    @staticmethod
    def calc_distance(pt1, pt2):
        """计算两点之间的欧氏距离和中点。"""
        length = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        center = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
        return length, center

    def is_big_armor(self, pts):
        """判断是否为大装甲板。

        通过比较灯条中心距与平均灯条长度的比值来判断。
        假设 pts 顺序为：左灯条底部、左灯条顶部、右灯条顶部、右灯条底部。
        """
        length1, center1 = self.calc_distance(pts[0], pts[1])
        length2, center2 = self.calc_distance(pts[3], pts[2])

        avg_light_length = (length1 + length2) / 2
        center_distance = (
            math.sqrt(
                (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
            )
            / avg_light_length
        )

        return center_distance > 3.2

    def process_roi(self, raw_img, pts):
        """通过透视变换提取装甲板数字区域。

        Args:
            raw_img: 原始图像
            pts: 装甲板四个顶点坐标

        Returns:
            二值化后的数字区域图像，失败时返回 None
        """
        light_length = 12
        warp_height = 28
        small_armor_width = 32
        large_armor_width = 54
        roi_size = (20, 28)

        top_light_y = (warp_height - light_length) // 2 - 1
        bottom_light_y = top_light_y + light_length
        warp_width = large_armor_width if self.is_big_armor(pts) else small_armor_width

        target_vertices = np.array(
            [
                [0, bottom_light_y],
                [0, top_light_y],
                [warp_width - 1, top_light_y],
                [warp_width - 1, bottom_light_y],
            ],
            dtype=np.float32,
        )

        # 确保传入的源点是 4 个
        src_pts = pts[:4] if len(pts) >= 4 else pts
        if len(src_pts) != 4:
            print(f"Error: src_pts length is {len(src_pts)}, expected 4.")
            return None

        src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 2)

        try:
            rotation_matrix = cv2.getPerspectiveTransform(src_pts, target_vertices)
            number_image = cv2.warpPerspective(
                raw_img, rotation_matrix, (warp_width, warp_height)
            )

            # 获取 ROI
            x = (warp_width - roi_size[0]) // 2
            y = 0
            number_image = number_image[y:y + roi_size[1], x:x + roi_size[0]]

            # 二值化
            number_image = cv2.cvtColor(number_image, cv2.COLOR_RGB2GRAY)
            _, number_image = cv2.threshold(
                number_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )

            return number_image
        except cv2.error as e:
            print(f"Error in getPerspectiveTransform: {e}")
            return None

    def draw_detections(self, frame, detections, color=(0, 255, 0)):
        """在图像上绘制检测框，同时进行数字识别。"""
        for d in detections:
            pts = np.array(d["position"], dtype=np.int32)
            # roi 点的顺序是：左下、左上、右上、右下
            pts2 = [pts[1], pts[0], pts[3], pts[2]]
            roi = self.process_roi(frame, pts2)

            num_id = ""
            if roi is not None:
                test = self.number.predict(roi)
                num_id = self.number_id[test["class_index"]]
                cv2.imshow("ROI", roi)

            cv2.putText(
                frame,
                f"Detections: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.polylines(frame, [pts], True, color, 2)  # type: ignore[arg-type]
            cv2.putText(
                frame,
                f"{d['cls']}{num_id} {d['conf']}",
                (pts[0][0], pts[0][1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return frame


# 向后兼容别名
infer2 = Infer2
