import logging
from utils import run_subprocess,order_preproc,crop_det_img, path_exist
from config import FILE_DICT, CONFIG

def detect_text(det_result_file, debug=False):
    run_subprocess(f"python3 {CONFIG['DET_SCRIPT']} -c {CONFIG['DET_CONFIG']} ",
                   desc="Text Detection",
                   check_output=det_result_file,
                   debug=debug)

def detect_order(ori_img_path,det_result_file,order_preprocess_file,order_result_file, debug=False):
    #preprocess order detection
    order_preproc(ori_img_path,det_result_file,order_preprocess_file)
    #check preprocess file exist
    if not path_exist(order_preprocess_file):
        logging.error(f"Expected output file missing: {order_preprocess_file}")
        raise RuntimeError(f"Order detection preprocess failed")
    # run order detection
    run_subprocess(f"bash {CONFIG['ORDER_SCRIPT']} --image_path {ori_img_path}",
    desc="Order Detection", check_output=order_result_file,debug=debug)
    
def recognize_text(ori_img_path,det_result_file,rec_result_file,debug=False):
    rec_preprocess = crop_det_img(ori_img_path,det_result_file)
    run_subprocess(
    f"python3 {CONFIG['REC_SCRIPT']} -c {CONFIG['REC_CONFIG']}",
    desc="Test Recognition",
    check_output=rec_result_file,debug=debug)