import os
import cv2
import torch
import onnxruntime
import numpy as np
from utils import poly_postprocess, vis,min_rect,ValTransform,demo_postprocess_armor,demo_postprocess_buff
from datasets import COCO_CLASSES
import time

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


    def visual(self, output, img_info, cls_conf=0.35):
            ratio = img_info["ratio"]
            img = img_info["raw_img"]
            if output is None:
                return img
            output = output.cpu()

            bboxes = output[:, 0:self.num_apexes*2]
            # preprocessing: resize
            bboxes /= ratio

            cls = output[:, self.num_apexes*2 + 2]
            scores = output[:, self.num_apexes*2] * output[:, self.num_apexes*2 + 1]
            # print(bboxes)
            # print(scores)
            # print(cls)
            vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
            return vis_res


if __name__ == "__main__":

    video_path = "./video/3.mp4"
    #onnx_model_path = "./model/500.onnx"
    onnx_model_path = "./model/opt-0625-001.onnx"
    #根据自己模型的不同可对关键点数量，颜色数量，类别数量进行相对应的修改
    predictor = Predictor(onnx_model_path, num_apex = 4, num_class = 8,num_color = 4)

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        outputs, img_info = predictor.inference(frame)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)

        cv2.imshow("Video", result_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 键退出循环
            break

    cap.release()
    cv2.destroyAllWindows()



