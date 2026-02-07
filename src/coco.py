import cv2
import numpy as np
from ultralytics import YOLO


def load_coco_seg_model(model_name="yolov8s-seg.pt"):
    """
    YOLOv8 COCO pretrained segmentation model
    """
    return YOLO(model_name)


def segment_image(model, image, conf=0.4):
    results = model.predict(source=image, task="segment", conf=conf, verbose=False)
    return results[0]


def hard_coco_mask(image_shape, result, allowed_classes):
    """
    Create HARD color mask (no transparency, no original image)
    """

    h, w = image_shape[:2]
    mask_img = np.zeros((h, w, 3), dtype=np.uint8)  # black background

    if result.masks is None:
        return mask_img

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    # Fixed colors for stability
    CLASS_COLORS = {
        0: (255, 0, 0),  # person
        1: (0, 255, 0),  # bicycle
        2: (0, 0, 255),  # car
        3: (255, 255, 0),  # motorcycle
        5: (255, 0, 255),  # bus
        7: (0, 255, 255),  # truck
        9: (128, 128, 0),  # traffic light
        11: (128, 0, 128),  # stop sign
    }

    for mask, cls_id in zip(masks, classes):

        if cls_id not in allowed_classes:
            continue

        binary_mask = mask > 0.5
        mask_img[binary_mask] = CLASS_COLORS[cls_id]

    return mask_img


def coco_road_segmentation(image, model_name="yolov8s-seg.pt"):

    COCO_ROAD_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        9: "traffic_light",
        11: "stop_sign",
    }

    model = load_coco_seg_model(model_name)
    result = segment_image(model, image)

    hard_mask = hard_coco_mask(
        image_shape=image.shape, result=result, allowed_classes=COCO_ROAD_CLASSES
    )

    hard_mask = cv2.cvtColor(hard_mask, cv2.COLOR_RGB2BGR)
    return hard_mask
