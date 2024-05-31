import random
import cv2
import argparse
from ultralytics import YOLO

def process_frame(frame, model1, model2, classlist1, classlist2, colors):
    output_width = 500
    output_height = 500

    frame = cv2.resize(frame, (output_width, output_height))

    parameters1 = model1.predict(source=[frame], conf=0.30, save=False)
    DP1 = parameters1[0].numpy()

    parameters2 = model2.predict(source=[frame], conf=0.30, save=False)
    DP2 = parameters2[0].numpy()

    if len(DP1) != 0:
        for i in range(len(parameters1[0])):
            boxes = parameters1[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            bb[0] *= output_width / frame.shape[1]
            bb[1] *= output_height / frame.shape[0]
            bb[2] *= output_width / frame.shape[1]
            bb[3] *= output_height / frame.shape[0]

            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), colors[int(clsID)], 3)

            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                classlist1[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    if len(DP2) != 0:
        for i in range(len(parameters2[0])):
            boxes = parameters2[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            bb[0] *= output_width / frame.shape[1]
            bb[1] *= output_height / frame.shape[0]
            bb[2] *= output_width / frame.shape[1]
            bb[3] *= output_height / frame.shape[0]

            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), colors[int(clsID)], 3)

            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                classlist2[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument("-i", "--input", required=True, help="path to input image, video, or 'live' for live detection")
    args = vars(parser.parse_args())


    coco_weights_path = "C:\\Users\\Bharathi\\yolov8\\weights\\yolov8n.pt"
    custom_weights_path = "C:\\Users\\Bharathi\\yolov8\\dataset\\runs\\detect\\train38\\best.pt"


    coco_textfile_path = "C:\\Users\\Bharathi\\yolov8\\utils\\coco.txt"
    custom_textfile_path = "C:\\Users\\Bharathi\\yolov8\\utils\\data.txt"

    with open(coco_textfile_path, "r") as coco_file:
        coco_names = coco_file.read().split("\n")
    with open(custom_textfile_path, "r") as custom_file:
        custom_names = custom_file.read().split("\n")


    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(max(len(coco_names), len(custom_names)))]


    coco_model = YOLO(coco_weights_path, "v8")
    custom_model = YOLO(custom_weights_path, "v8")

    if args["input"].lower() == "live":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            exit()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            processed_frame = process_frame(frame, coco_model, custom_model, coco_names, custom_names, colors)
            cv2.imshow("Live Object Detection", processed_frame)
            if cv2.waitKey(1) == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    else:
        if args["input"].lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            frame = cv2.imread(args["input"])
            if frame is None:
                print("Error: Unable to load the image.")
            else:
                processed_frame = process_frame(frame, coco_model, custom_model, coco_names, custom_names, colors)
                cv2.imshow("Object Detection", processed_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        elif args["input"].lower().endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(args["input"])
            if not cap.isOpened():
                print("Cannot open video file")
                exit()
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                processed_frame = process_frame(frame, coco_model, custom_model, coco_names, custom_names, colors)
                cv2.imshow("Object Detection", processed_frame)
                if cv2.waitKey(1) == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()

        else:
            print("Unsupported file format or 'live' option not specified.")
