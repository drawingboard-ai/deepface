from deepface import DeepFace

import base64
from tempfile import NamedTemporaryFile
import os

# pylint: disable=broad-except


def represent(img_path, model_name, detector_backend, enforce_detection, align):
    try:
        result = {}
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
        )
        result["results"] = embedding_objs
        return result
    except Exception as err:
        return {"error": f"Exception while representing: {str(err)}"}, 400


def verify(
    img1_path, img2_path, model_name, detector_backend, distance_metric, enforce_detection, align
):
    try:
        obj = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
        )
        return obj
    except Exception as err:
        return {"error": f"Exception while verifying: {str(err)}"}, 400


def analyze(base64_image, actions, detector_backend, enforce_detection, align):
    try:
        result = {}

        image_data = base64.b64decode(base64_image)

        with NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(image_data)
            tmp_file_path = tmp_file.name

        demographies = DeepFace.analyze(
            img_path=tmp_file_path,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            silent=True,
        )

        result["results"] = demographies

        os.remove(tmp_file_path)

        return result
    except Exception as err:
        if 'tmp_file_path' in locals():
            os.remove(tmp_file_path)
        return {"error": f"Exception while analyzing: {str(err)}"}, 400
