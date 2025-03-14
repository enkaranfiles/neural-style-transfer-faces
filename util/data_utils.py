import cv2
import dlib
import numpy as np

def create_beard_mask_custom(image_path: str, predictor_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")
    h, w, _ = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    faces = detector(gray)
    if not faces:
        raise ValueError("No faces detected in the image.")
    face = faces[0]
    shape = predictor(gray, face)

    points = [(shape.part(i).x, shape.part(i).y) for i in range(1, 16)]
    mask_e = [(shape.part(i).x, shape.part(i).y) for i in [35, 34, 33, 32, 31]]
    fmask_e = points + mask_e

    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(fmask_e, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)

    mouth_points = [(shape.part(i).x, shape.part(i).y) for i in range(48, 68)]
    mouth_pts = np.array(mouth_points, dtype=np.int32).reshape((-1, 1, 2))
    mouth_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mouth_mask, [mouth_pts], 255)
    mask[mouth_mask == 255] = 0

    return mask