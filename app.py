import gradio as gr
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


def detect_age_gender(image):
    new_image = image.copy()
    fd_output = fd_model(image, verbose=False)

    results = Detections.from_ultralytics(fd_output[0])

    for ind, result in enumerate(results):
        if len(results.xyxy[ind]) > 0:
            bbox = [int(b) for b in results.xyxy[ind]]  # xyxy = [x1, y1, x2, y2]
            confidence = results.confidence[ind]
            face = crop_face(image, bbox)
            gender = detect_age_or_gender(face, gd_model)
            age = detect_age_or_gender(face, ad_model)
            new_image = cv2.rectangle(new_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (36, 255, 12), 3)
            cv2.putText(new_image,
                        f"{gender}, {age} ({confidence:.2f})",
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.59, (36, 255, 12), 2)

    return new_image


def init_models():
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

    return fd_model, gd_model, ad_model


fd_model, gd_model, ad_model = init_models()


# Gradio Interface
with gr.Blocks(theme=gr.themes.Glass()) as demo:
    gr.Markdown("# Age and Gender Detection App")
    gr.Markdown("Click a button to either upload an image or use your camera for real-time detection.")

    with gr.Tab("Single Image"):
        gr.Interface(fn=detect_age_gender,
                     inputs=gr.Image(type="numpy", label="Input Image", width="100%", height=640),
                     outputs=gr.Image(type="numpy", label="Output Image", width="100%", height=640))

    with gr.Tab("Video Capture"):
        # The webcam image component. The `sources=["webcam"]` and `streaming=True`
        # are the key components for real-time video processing.
        webcam_input = gr.Image(
            sources=["webcam"],
            streaming=True,
            interactive=True,
            type="numpy",
            label="Webcam Input",
            width="100%",
            height=480
        )

        # The output component where the processed video will be displayed.
        output_image = gr.Image(
            label="Processed Feed",
            width="100%",
            height=480
        )

        # The `stream()` method automatically calls the `process_frame` function
        # for each frame of the video feed.
        webcam_input.stream(detect_age_gender,
                            inputs=webcam_input,
                            outputs=output_image)

demo.launch()