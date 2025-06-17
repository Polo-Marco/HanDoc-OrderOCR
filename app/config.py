import os

DEBUG = False
FILE_DICT = {
    "SAVE_FOLDER": os.path.abspath("./saved/"),
    "REC_FOLDER": os.path.abspath("../output/rec_images/"),
    "ORDER_FOLDER": os.path.abspath("../output/order_results/"),
    "PROCESSED_FOLDER": os.path.abspath("./processed/"),
}
CONFIG = {
    "DET_CONFIG": "../configs/det_r50_dbpp_mth.yml",
    "REC_CONFIG": "../configs/mth_rec_svtrnet.yml",
    "ORDER_SCRIPT": "../scripts/order_predict.sh",
    "CLEAN_SCRIPT": "../scripts/clean.sh",
    "DET_SCRIPT": "../PaddleOCR/tools/infer_det.py",
    "REC_SCRIPT": "../PaddleOCR/tools/infer_rec.py",
}
RESULT_FILES = {
    "DET_RESULT": "../output/det_results/predicts.txt",
    "ORDER_PREPROCESS": "../output/order_results/predict.json",
    "ORDER_RESULT": "../output/order_results/predicted.json",
    "REC_RESULT": "../output/rec_results/predicts.txt",
}
