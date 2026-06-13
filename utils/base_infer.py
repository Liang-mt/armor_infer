import torch
import numpy as np
from utils import poly_postprocess, min_rect, ValTransform, demo_postprocess_armor, demo_postprocess_buff
from utils.utils import create_session


class BaseInfer(object):
    """推理基类，封装 YOLOX 模型的公共推理流程。

    子类需实现 detect() 和 draw_detections() 方法。
    """

    def __init__(
        self,
        onnx_model_path,
        num_apex,
        num_class,
        num_color,
        device="cpu",
        legacy=False,
    ):
        self.session = create_session(onnx_model_path)
        self.num_apexes = num_apex
        self.num_classes = num_class
        self.num_colors = num_color
        self.confthre = 0.25
        self.nmsthre = 0.3
        self.test_size = (416, 416)
        self.device = device
        self.preproc = ValTransform(legacy=legacy)

        # 缓存 ONNX 输入输出名，避免每次推理重复查询
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

        # 颜色和类别 ID 映射
        self.color_id = ["B", "R", "N", "P"]
        self.cls_id = ["B", "1", "2", "3", "4", "5", "G", "O", "base"]

    def inference(self, img):
        """完整的推理流程：预处理 → ONNX 推理 → 解码 → NMS。"""
        img_info = {}

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float().numpy()

        outputs = self.session.run(
            [self._output_name], input_feed={self._input_name: img}
        )

        if self.num_apexes == 4:
            outputs = demo_postprocess_armor(outputs[0], self.test_size, p6=False)[0]
        elif self.num_apexes == 5:
            outputs = demo_postprocess_buff(outputs[0], self.test_size, p6=False)[0]

        # 将多边形顶点转换为 [cx, cy, w, h] 格式的 bbox
        bbox_preds = []
        for i in range(outputs.shape[0]):
            bbox = min_rect(outputs[i, :, :self.num_apexes * 2])
            bbox_preds.append(bbox)
        bbox_preds = torch.stack(bbox_preds)

        conf_preds = outputs[:, :, self.num_apexes * 2].unsqueeze(-1)

        # 组合颜色和类别预测
        cls_preds = outputs[:, :, self.num_apexes * 2 + 1 + self.num_colors:].repeat(
            1, 1, self.num_colors
        )
        colors_preds = torch.clone(cls_preds)

        for i in range(self.num_colors):
            colors_preds[
                :, :, i * self.num_classes : (i + 1) * self.num_classes
            ] = outputs[
                :, :,
                self.num_apexes * 2 + 1 + i : self.num_apexes * 2 + 1 + i + 1,
            ].repeat(
                1, 1, self.num_classes
            )

        cls_preds_converted = (colors_preds + cls_preds) / 2.0

        outputs_rect = torch.cat(
            (bbox_preds, conf_preds, cls_preds_converted), dim=2
        )
        outputs_poly = torch.cat(
            (outputs[:, :, : self.num_apexes * 2], conf_preds, cls_preds_converted),
            dim=2,
        )

        outputs = poly_postprocess(
            outputs_rect,
            outputs_poly,
            self.num_apexes,
            self.num_classes * self.num_colors,
            self.confthre,
            self.nmsthre,
        )
        return outputs, img_info

    def _parse_position(self, d):
        """从检测结果中解析顶点坐标数组。

        Args:
            d: 包含坐标值的张量，长度为 num_apexes * 2

        Returns:
            np.ndarray: shape (num_apexes, 2) 的 float32 坐标数组
        """
        pts = [(int(d[2 * j]), int(d[2 * j + 1])) for j in range(self.num_apexes)]
        return np.array(pts, dtype=np.float32).reshape(self.num_apexes, 2)

    def get_color_and_tag(self, label):
        """根据组合类别索引解码出颜色和标签索引。"""
        color = label // 9
        tag = label % 9
        return color, tag
