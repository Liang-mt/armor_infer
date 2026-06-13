from .utils import (
    create_session,
    poly_postprocess,
    vis,
    min_rect,
    ValTransform,
    demo_postprocess,
    demo_postprocess_armor,
    demo_postprocess_buff,
)
from .base_infer import BaseInfer
from .infer import Infer, infer
from .infer2 import Infer2, infer2
from .mlp_predict import NumberClassifier, number_cls
from .datasets import (
    COCO_CLASSES_buff,
    COCO_CLASSES_rba,
    COCO_CLASSES_long,
    COCO_CLASSES_tup,
)

# 向后兼容：main2.py 中 from utils import COCO_CLASSES 的引用
COCO_CLASSES = COCO_CLASSES_tup

__all__ = [
    # 工具函数
    "create_session",
    "poly_postprocess",
    "vis",
    "min_rect",
    "ValTransform",
    "demo_postprocess",
    "demo_postprocess_armor",
    "demo_postprocess_buff",
    # 基类
    "BaseInfer",
    # 推理类（新名 + 向后兼容别名）
    "Infer",
    "infer",
    "Infer2",
    "infer2",
    # 分类器（新名 + 向后兼容别名）
    "NumberClassifier",
    "number_cls",
    # 数据集类名
    "COCO_CLASSES",
    "COCO_CLASSES_buff",
    "COCO_CLASSES_rba",
    "COCO_CLASSES_long",
    "COCO_CLASSES_tup",
]
