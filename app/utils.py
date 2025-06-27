import json
import logging
import os
import subprocess
import warnings

import cv2
import numpy as np
from config import FILE_DICT
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

warnings.filterwarnings("ignore")
font = ImageFont.truetype("./font/NotoSerifCJKtc-ExtraLight.otf", 30)


def path_exist(path) -> bool:
    return True if os.path.exists(path) else False


def run_subprocess(cmd, desc=None, check_output=None, debug=True):
    """
    Runs a subprocess command with logging and error handling.

    Args:
        cmd (str): Command to execute.
        desc (str): Description for logging.
        check_output (str): Path to output file to check existence after run.
        debug: for debugging subprocess
    Returns:
        CompletedProcess object.
    Raises:
        RuntimeError if command fails or output file does not exist.
    """
    if desc:
        logging.info(f"Starting: {desc}")
    else:
        logging.info(f"Running: {cmd}")
    if debug:
        completed = subprocess.run(cmd, shell=True)
    else:
        completed = subprocess.run(
            cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    if completed.returncode != 0:
        logging.error(f"Subprocess failed: {cmd}")
        raise RuntimeError(
            f"{desc or cmd} failed with return code {completed.returncode}"
        )
    if check_output and not path_exist(check_output):
        logging.error(f"Expected output file missing: {check_output}")
        raise RuntimeError(f"{desc or cmd} failed with no output file {check_output}")
    logging.info(f"Subprocess completed: {cmd}")
    return completed


def crop_det_img(img_path, det_result_file) -> None:
    logging.info("Cropping detected text boxes")
    try:
        det_anno = read_det(det_result_file)
        img = cv2.imread(img_path)
        for idx, anno in enumerate(det_anno):
            crop_name = f"{img_path.split('/')[-1][:-4]}_{idx}.jpg"
            cropped_img = crop_img(img, anno["points"], rotate=270)
            out_path = os.path.join(FILE_DICT["REC_FOLDER"], crop_name)
            cv2.imwrite(out_path, cropped_img)
        logging.info("Recognition preprocess completed")
    except Exception as error:
        logging.error(f"Recognition preprocess failed with error: {error}")
        raise RuntimeError(f"Recognition preprocess failed with error: {error}")


def crop_img(img, poly_pts, pad_color="white", rotate=None):
    # mapping to cv2 rotate code
    rotate_map = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    pts = np.array(poly_pts, dtype=np.int32)
    # (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    cropped = img[y : y + h, x : x + w].copy()
    # (2) make mask
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # (3) do bit-op
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
    # (4) add the white background
    bg = np.ones_like(cropped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst
    if pad_color == "white":
        if rotate:
            return cv2.rotate(dst2, rotate_map[rotate])
        return dst2
    if rotate:
        return cv2.rotate(dst, rotate_map[rotate])
    return dst


# visualize det & rec
def read_det(det_path):
    with open(det_path, "r") as f:
        return json.loads(f.readline().split("\t")[1])


def read_rec(rec_path):
    rec_anno = {}
    with open(rec_path, "r") as f:
        line = f.readline()
        while line:
            line = line.split("\t")
            file, anno = line[0][:-4], line[1]
            file = file.split("_")[-1]
            rec_anno[file] = anno
            line = f.readline()
    return rec_anno


def read_order(order_path):
    """Return mapping from detection index to predicted order index."""
    with open(order_path, "r") as f:
        data = json.load(f)

    # predicted.json only contains a single entry with the image name as the key
    # and a list indicating the reading order of detection indices
    orders = list(data.values())[0]

    # convert to {det_idx: order_idx}
    return {int(det_idx): order_idx for order_idx, det_idx in enumerate(orders)}


def return_tl_rb(boxes):
    xs = [box[0] for box in boxes]
    ys = [box[1] for box in boxes]
    return (min(xs), min(ys)), (max(xs), max(ys))


def vis_det_rec(ori_img_path, det_path, rec_path, order_path):
    rec_anno = read_rec(rec_path)
    det_anno = read_det(det_path)
    order_anno = read_order(order_path)
    image = Image.open(ori_img_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGB")
    for idx, annotation in enumerate(det_anno):
        poly = [
            elem for elem1 in annotation["points"] for elem in elem1
        ]  # flat 2d cordinates to 1d
        tl, br = return_tl_rb(annotation["points"])
        poly_outline = (255, 0, 0)
        # Display the predicted order index followed by the recognized text
        text = f"{order_anno.get(idx, '?')}: {rec_anno.get(str(idx), '')}"
        # poly_outline=(255,0,0)
        poly_center = (
            tl[0] + (abs(tl[0] - br[0])) / 2,
            tl[1] + abs((tl[1] - br[1])) / 2,
        )
        poly_center = (poly_center[0] + (abs(tl[0] - br[0]) / 2), poly_center[1])
        # Draw sentence boxes
        draw.polygon(poly, outline=poly_outline, width=3)
        l_u_point = poly_center
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        else:
            w, h = draw.textsize(text, font=font)
        draw.rectangle(
            (
                l_u_point[0] - w / 2,
                l_u_point[1] - h / 2,
                l_u_point[0] + w / 2,
                l_u_point[1] + h / 2,
            ),
            fill=(64, 64, 64, 0),
        )
        draw.text(
            (l_u_point[0] - w / 2, l_u_point[1] - h / 2),
            text=text,
            fill=(255, 255, 255, 255),
            font=font,
        )
    return image


def get_seq_text(det_path, rec_path, order_path):
    rec_anno = read_rec(rec_path)
    det_anno = read_det(det_path)
    order_anno = read_order(order_path)
    seq_text = list(range(len(order_anno.keys())))
    for idx, annotation in enumerate(det_anno):
        rec_text = rec_anno.get(str(idx), "")
        seq_text[order_anno.get(idx, 0)] = rec_text
    return seq_text


def order_preproc(img_path, det_result_path, preprocessed_file):
    try:
        det_anno = {}
        filename = img_path.split("/")[-1][:-4]
        height, width = cv2.imread(img_path).shape[:2]
        det_anno[filename] = {
            "line_label": [i["points"] for i in read_det(det_result_path)],
            "width": width,
            "height": height,
        }
        if len(det_anno[filename]["line_label"]) == 0:
            raise ValueError(
                "No text lines were detected in the image. "
                "Possible reasons: the detection model failed, "
                "or there may be no recognizable text in the uploaded image. "
                "Tip: For best results, please upload historical Chinese documents or use provided examples"
            )
        with open(preprocessed_file, "w") as f:
            json.dump(save_pair(pair_data(det_anno)), f)
    except Exception as error:
        logging.error(f"Order preprocess failed with error: {error}")
        raise RuntimeError(f"Order preprocess failed with error: {error}")


def pair_data(data_dict):
    """
    input: imgs dict
    output: pair-wised dataset with sentence and text as key
    """
    sent_pair = {}
    for img in tqdm(data_dict.keys()):
        w, h = data_dict[img]["width"], data_dict[img]["height"]
        features, pairs = pair(data_dict[img]["line_label"], w, h)
        sent_pair[img] = {"features": features, "index": pairs, "width": w, "height": h}
    return sent_pair


def flatten_lst(seq):
    return [seq_1d for seq_2d in seq for seq_1d in seq_2d]


def pair(seq, width, height):
    re_lst = []
    pair = []
    if width is None:
        width = 1
    if height is None:
        height = 1
    for i in range(len(seq)):
        for j in range(len(seq)):
            if i == j:
                continue
            feature = flatten_lst(seq[i]) + flatten_lst(seq[j])
            normalized_feature = [
                elem / width if idx % 2 == 0 else elem / height
                for idx, elem in enumerate(feature)
            ]
            re_lst.append(normalized_feature)
            pair.append([i, j])
    return re_lst, pair


def save_pair(data):
    re_lst = []
    for img in data:
        for feature, idx in zip(data[img]["features"], data[img]["index"]):
            re_lst.append(
                {
                    "feature": feature,
                    "label": 0,
                    "index": idx,
                    "image": img,
                    "width": data[img]["width"],
                    "height": data[img]["height"],
                }
            )
    return re_lst


if __name__ == "__main__":
    print("end utils")
