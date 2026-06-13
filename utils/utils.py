import os
import sys

import torch
import torchvision
import numpy as np
import cv2
import onnxruntime as ort

# 将 PyTorch 的 CUDA/cuDNN DLL 路径加入 PATH，让 ONNX Runtime 能找到
_torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if _torch_lib not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _torch_lib + ';' + os.environ.get('PATH', '')


def create_session(onnx_model_path, force_cpu=False):
    """创建 ONNX 推理会话，优先使用 GPU（CUDA），失败则静默回退 CPU。

    Args:
        onnx_model_path: ONNX 模型文件路径
        force_cpu: 强制使用 CPU（适用于小模型，GPU 反而更慢）

    Returns:
        ort.InferenceSession: 已初始化的推理会话
    """
    if force_cpu:
        session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider'],
        )
        print(f"🖥️  Provider: CPUExecutionProvider")
        return session

    # 尝试 CUDA，抑制错误输出
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        session = ort.InferenceSession(
            onnx_model_path,
            providers=['CUDAExecutionProvider'],
            provider_options=[{'device_id': 0}],
        )
        sys.stderr.close()
        sys.stderr = old_stderr
        if 'CUDAExecutionProvider' in session.get_providers():
            print(f"🖥️  Provider: CUDAExecutionProvider")
            return session
    except Exception:
        pass
    finally:
        try:
            sys.stderr.close()
        except Exception:
            pass
        sys.stderr = old_stderr

    # 回退 CPU
    session = ort.InferenceSession(
        onnx_model_path,
        providers=['CPUExecutionProvider'],
    )
    print(f"🖥️  Provider: CPUExecutionProvider")
    return session

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

def preproc(img, input_size, swap=(2, 0, 1)):
    """
    Preprocess input images.

    Resize and transpose image format from (y,x,channels) to (channels,x,y)
    """
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114
    # Caclute resize matrix
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # Transpose to (channels,x,y)
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

class ValTransform:

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))


def min_rect(apex):
    """
    Generate [cx,cy,w,h] format rect bbox for apexs.
    """
    x = apex[:,::2]
    y = apex[:,1::2]

    x_min = torch.min(x, dim=1, keepdim=True).values
    x_max = torch.max(x, dim=1, keepdim=True).values
    y_min = torch.min(y, dim=1, keepdim=True).values
    y_max = torch.max(y, dim=1, keepdim=True).values

    w = x_max - x_min
    h = y_max - y_min

    rect_bbox = torch.cat(((x_min + w / 2),
                            (y_min + h / 2),
                            w,
                            h), dim=1)
    return rect_bbox

def poly_postprocess(prediction_rect, prediction_poly, num_apex, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction_rect.new(prediction_rect.shape)
    box_corner[:, :, 0] = prediction_rect[:, :, 0] - prediction_rect[:, :, 2] / 2
    box_corner[:, :, 1] = prediction_rect[:, :, 1] - prediction_rect[:, :, 3] / 2
    box_corner[:, :, 2] = prediction_rect[:, :, 0] + prediction_rect[:, :, 2] / 2
    box_corner[:, :, 3] = prediction_rect[:, :, 1] + prediction_rect[:, :, 3] / 2
    prediction_rect[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction_rect))]
    for i, image_pred in enumerate(prediction_rect):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)


        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections_Rect ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections_rect = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections_poly = torch.cat((prediction_poly[i,:,:2 * num_apex + 1], class_conf, class_pred.float()), 1)
        detections_rect = detections_rect[conf_mask]
        detections_poly = detections_poly[conf_mask]
        if not detections_rect.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections_rect[:, :4],
                detections_rect[:, 4] * detections_rect[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections_rect[:, :4],
                detections_rect[:, 4] * detections_rect[:, 5],
                detections_rect[:, 6],
                nms_thre,
            )

        detections = detections_poly[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    rect_bbox = min_rect(boxes)
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        # print(box)
        x0 = int(rect_bbox[i][0])
        y0 = int(rect_bbox[i][1])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        num_pts = int(len(box) / 2)
        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        pts = []
        for j in range(num_pts):
            pts.append(tuple(np.array(box[(2 * j):(2 * (j + 1))], dtype=np.int32)))
        for j in range(num_pts):
            cv2.line(img, pts[j], pts[(j + 1) % num_pts], color, 2)

        # cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img



def demo_postprocess(predictions, img_size, num_apex, p6=False):
    """通用的 YOLOX 后处理函数，支持不同数量的 apex 点。

    Args:
        predictions: 原始网络输出
        img_size: 输入图像尺寸 (h, w)
        num_apex: 关键点数量（4=装甲板，5=能量机关）
        p6: 是否使用 P6 stride
    """
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        xv, yv = xv.astype(np.float32), yv.astype(np.float32)
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride, dtype=np.float32))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)

    predictions = np.expand_dims(predictions, axis=0)

    # 解析每个 apex 的坐标并调整
    num_coords = num_apex * 2
    splits = np.split(predictions[..., :num_coords], num_apex, axis=-1)
    adjusted = [(s + grids) * expanded_strides for s in splits]

    adjusted_outputs = np.concatenate(
        [*adjusted, predictions[..., num_coords:]], axis=-1
    )

    return torch.tensor(adjusted_outputs, dtype=torch.float32)


def demo_postprocess_armor(predictions, img_size, p6=False):
    """装甲板检测后处理（4个关键点）。向后兼容包装函数。"""
    return demo_postprocess(predictions, img_size, num_apex=4, p6=p6)


def demo_postprocess_buff(predictions, img_size, p6=False):
    """能量机关检测后处理（5个关键点）。向后兼容包装函数。"""
    return demo_postprocess(predictions, img_size, num_apex=5, p6=p6)