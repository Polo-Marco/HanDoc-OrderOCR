import logging

from config import CONFIG, RESULT_FILES
from utils import crop_det_img, order_preproc, path_exist, run_subprocess


def detect_text(debug=False) -> None:
    run_subprocess(
        f"python3 {CONFIG['DET_SCRIPT']} -c {CONFIG['DET_CONFIG']} ",
        desc="Text Detection",
        check_output=RESULT_FILES["DET_RESULT"],
        debug=debug,
    )


def detect_order(ori_img_path, debug=False) -> None:
    # preprocess order detection
    order_preproc(
        ori_img_path, RESULT_FILES["DET_RESULT"], RESULT_FILES["ORDER_PREPROCESS"]
    )
    # check preprocess file exist
    if not path_exist(RESULT_FILES["ORDER_PREPROCESS"]):
        logging.error(
            f"Expected output file missing: {RESULT_FILES['ORDER_PREPROCESS']}"
        )
        raise RuntimeError("Order detection preprocess failed")
    # run order detection
    run_subprocess(
        f"bash {CONFIG['ORDER_SCRIPT']}",
        desc="Order Detection",
        check_output=RESULT_FILES["ORDER_RESULT"],
        debug=debug,
    )


def recognize_text(ori_img_path, debug=False) -> None:
    crop_det_img(ori_img_path, RESULT_FILES["DET_RESULT"])
    run_subprocess(
        f"python3 {CONFIG['REC_SCRIPT']} -c {CONFIG['REC_CONFIG']}",
        desc="Text Recognition",
        check_output=RESULT_FILES["REC_RESULT"],
        debug=debug,
    )


def clean_temp(debug=False) -> None:
    run_subprocess(
        f"bash {CONFIG['CLEAN_SCRIPT']}",
        desc="Clean temp files",
        check_output=None,
        debug=debug,
    )
