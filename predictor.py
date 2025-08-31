import cv2
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections


def detect_age_or_gender(face, model):
    result = model(face, verbose=False)
    index_mapping = result[0].names

    result_index = result[0].probs.top1
    label = index_mapping[result_index]

    return label


def crop_face(img, bbox):
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def main():
    # ----------------------- CONFIGURATIONS ----------------------- #

    FACE_DETECTION_MODEL = "arnabdhar/YOLOv8-Face-Detection"
    GENDER_DETECTION_MODEL = str(Path("models") / "gender.pt")
    AGE_DETECTION_MODEL = str(Path("models") / "age.pt")

    # -------------------------------------------------------------- #

    # Load Face Detection (fd) Model
    model_path = hf_hub_download(repo_id=FACE_DETECTION_MODEL, filename="model.pt")
    fd_model = YOLO(model_path)

    # Load Gender Detection (gd) Model
    gd_model = YOLO(GENDER_DETECTION_MODEL)

    # Load Age Detection (ad) Model
    ad_model = YOLO(AGE_DETECTION_MODEL)

    # inference
    video = cv2.VideoCapture(0)
    while True:
        _, frame = video.read()

        fd_output = fd_model(frame, verbose=False)
        results = Detections.from_ultralytics(fd_output[0])

        for ind, result in enumerate(results):
            if len(results.xyxy[ind]) > 0:
                bbox = [int(b) for b in results.xyxy[ind]]            # xyxy = [x1, y1, x2, y2]
                confidence = results.confidence[ind]
                face = crop_face(frame, bbox)
                gender = detect_age_or_gender(face, gd_model)
                age = detect_age_or_gender(face, ad_model)
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (36, 255, 12), 3)
                cv2.putText(frame,
                            f"{gender}, {age} ({confidence:.2f})",
                            (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (36, 255, 12),2)

        cv2.imshow('Camera', frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    print("Finished")
