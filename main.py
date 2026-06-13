import time
import cv2
from utils import Infer2,Infer

if __name__ == "__main__":
    video_path = "./video/1.mp4"
    onnx_model_path = "./model/og1000.onnx"

    # 可根据模型切换推理类和参数：
    predictor = Infer(onnx_model_path, num_apex=4, num_class=9, num_color=4)
    # predictor = Infer(onnx_model_path, num_apex=5, num_class=2, num_color=2)
    #predictor = Infer2(onnx_model_path, num_apex=4, num_class=1, num_color=3)

    cap = cv2.VideoCapture(video_path)

    # Define the codec and create a VideoWriter object
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter("output_video.avi", fourcc, fps, (width, height))

    frame_count = 0
    total_detect_time = 0.0
    total_frame_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t_frame_start = time.perf_counter()

        t_detect_start = time.perf_counter()
        result_image, detections = predictor.detect(frame)
        t_detect_end = time.perf_counter()

        #out.write(result_image)
        cv2.imshow("Video", result_image)

        t_frame_end = time.perf_counter()

        detect_ms = (t_detect_end - t_detect_start) * 1000
        frame_ms = (t_frame_end - t_frame_start) * 1000
        total_detect_time += detect_ms
        total_frame_time += frame_ms
        frame_count += 1

        if frame_count % 30 == 0:
            avg_detect = total_detect_time / frame_count
            avg_frame = total_frame_time / frame_count
            print(
                f"帧 {frame_count:4d} | "
                f"检测 {detect_ms:.1f}ms | "
                f"总帧 {frame_ms:.1f}ms | "
                f"平均检测 {avg_detect:.1f}ms | "
                f"平均FPS {1000 / avg_frame:.1f}"
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if frame_count > 0:
        avg_detect = total_detect_time / frame_count
        avg_frame = total_frame_time / frame_count
        print(f"\n{'=' * 50}")
        print(f"总帧数: {frame_count}")
        print(f"平均检测耗时: {avg_detect:.1f}ms")
        print(f"平均总帧耗时: {avg_frame:.1f}ms")
        print(f"平均 FPS: {1000 / avg_frame:.1f}")
        print(f"瓶颈占比: 检测 {avg_detect / avg_frame * 100:.0f}% | 其他 {(1 - avg_detect / avg_frame) * 100:.0f}%")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
