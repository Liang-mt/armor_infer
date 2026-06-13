import numpy as np
import cv2
from utils.utils import create_session


class NumberClassifier(object):
    """基于 MLP 的装甲板数字分类器。

    输入来自 process_roi 输出的二值化图像（20x28），
    输出分类结果和置信度。
    """

    def __init__(self, model_path):
        # MLP 模型极小（20×28 输入），GPU 启动开销反而比计算慢，强制 CPU
        self.session = create_session(model_path, force_cpu=True)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, number_image):
        """预处理：类型转换、归一化、添加批次维度。"""
        if number_image.dtype != np.float32:
            number_image = number_image.astype(np.float32) / 255.0
        return number_image.reshape(1, 20, 28, 1)

    def predict(self, number_image):
        """执行推理并返回分类结果。

        Args:
            number_image: 20x28 的二值化图像

        Returns:
            dict: {"class_index": int, "confidence": float}

        Raises:
            ValueError: 输入图像为空时
        """
        if number_image is None or number_image.size == 0:
            raise ValueError("输入图像为空")

        input_data = self.preprocess(number_image)
        outputs = self.session.run(
            [self.output_name], {self.input_name: input_data}
        )

        pred_probs = outputs[0][0]
        predicted_class = np.argmax(pred_probs)

        return {
            "class_index": int(predicted_class),
            "confidence": float(pred_probs[predicted_class]),
        }


# 向后兼容别名
number_cls = NumberClassifier


# 使用示例
if __name__ == "__main__":
    recognizer = NumberClassifier("../model/mlp.onnx")

    # 模拟 process_roi 输出（20x28 二值化图像）
    dummy_image = np.random.randint(0, 255, (20, 28), dtype=np.uint8)
    _, test_image = cv2.threshold(dummy_image, 127, 255, cv2.THRESH_BINARY)

    result = recognizer.predict(test_image)
    print(f"预测结果: 类别{result['class_index']} 置信度{result['confidence']:.2f}")
