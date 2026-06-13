import cv2
import numpy as np
from utils.base_infer import BaseInfer


class Infer(BaseInfer):
    """单阶段推理类，用于装甲板/能量机关检测。

    使用组合类别索引（颜色×类型）进行检测结果解码。
    """

    def visual(self, output, img_info, conf=0.35):
        """解析检测结果为带颜色和类别标签的字典列表。"""
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
            color, cls = self.get_color_and_tag(cls_id)
            color = self.color_id[color]
            cls = self.cls_id[cls]

            score = round(scores[i].item(), 2)
            if score < conf:
                continue

            detections.append(
                {"color": color, "cls": cls, "conf": score, "position": position}
            )

        return detections

    def detect(self, frame):
        """对单帧进行检测，返回标注图像和检测结果。"""
        outputs, img_info = self.inference(frame)
        detections = self.visual(outputs[0], img_info, self.confthre)

        original_frame = self.draw_detections(frame.copy(), detections, (0, 255, 0))

        return original_frame, detections

    def draw_detections(self, frame, detections, color=(0, 255, 0)):
        """在图像上绘制检测框和标签。"""
        for d in detections:
            pts = np.array(d["position"], dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 2)  # type: ignore[arg-type]
            cv2.putText(
                frame,
                f"{d['color']}{d['cls']} {d['conf']}",
                (pts[0][0], pts[0][1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return frame


# 向后兼容别名
infer = Infer
