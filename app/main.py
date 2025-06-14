import gradio as gr
import numpy as np
from PIL import Image
import os
import subprocess
import json
from utils import *
import cv2
import threading
import time
file_dict = {"SAVE_FOLDER":"./saved/",
            "REC_FOLDER" : "../output/rec_images/",
            "ORDER_FOLDER":"../output/order_results/",
            "PROCESSED_FOLDER":"./processed/"}

def crop_det_img(filename,det_result_path):
    print("croping det results")
    det_anno = read_det(det_result_path)
    img = cv2.imread(file_dict['SAVE_FOLDER']+filename)
    for idx,anno in enumerate(det_anno):
        cropped_img = crop_img(img,anno["points"],rotate=270)
        cv2.imwrite(file_dict['REC_FOLDER']+filename[:-4]+"_"+str(idx)+".jpg",cropped_img)
    print("rec preprocess completed")
    return None
import time#for debug
def process_image(input_image,progress):
    filename = "input_image.jpg"
    #save image first
    progress(0.1, desc="Saving image")
    #image = Image.fromarray((input_image * 255).astype(np.uint8))
    input_image.save(f"{file_dict['SAVE_FOLDER']}input_image.jpg")

    #detection
    progress(0.2, desc="Detecting texts")
    completed_process = subprocess.run(f"python3 ../PaddleOCR/tools/infer_det.py -c ../configs/det_r50_dbpp_mth.yml", 
                                       shell=True)
    if completed_process.returncode != 0:
            raise RuntimeError("Detection failed.")
    #order detection
    progress(0.5, desc="Detecting order")
    order_preproc(file_dict['SAVE_FOLDER'],[filename],"../output/det_results/predicts.txt",file_dict['ORDER_FOLDER'])
    order_thread = threading.Thread(target = subprocess.run,args=("bash ../scripts/order_predict.sh",), 
                                              kwargs={'shell': True})
    order_thread.start()
    order_thread.join()
    print("order process complete")
    progress(0.7, desc="Recognizing texts")
    ## rec preprocess start
    rec_preproc_thread = threading.Thread(target = crop_det_img, args = (filename,"../output/det_results/predicts.txt"))
    rec_preproc_thread.start()
    ## rec_process
    rec_preproc_thread.join()#make sure crop process done
    completed_process = subprocess.run("python3 ../PaddleOCR/tools/infer_rec.py -c ../configs/mth_rec_svtrnet.yml",shell=True)
    if completed_process.returncode == 0:
        print("rec process success")
    else:
        print("rec process failed")

    progress(0.8, desc="Visualizing results")
    #visualize det + rec results
    img_vis,seq_txt=vis_det_rec(filename,
                        file_dict['SAVE_FOLDER']+filename,"../output/det_results/predicts.txt",
                        "../output/rec_results/predicts.txt",
                        file_dict['ORDER_FOLDER'])
    img_vis.save(file_dict['PROCESSED_FOLDER']+filename)
    return seq_txt

def gradio_interface(input_image,progress=gr.Progress()):
    text_output = process_image(input_image,progress)
    processed_img_path = file_dict['PROCESSED_FOLDER']+"input_image.jpg"
    with Image.open(processed_img_path) as img:
        output_image = img.copy()
    # 處理完後，將處理結果顯示在網頁上
    subprocess.run("bash ../scripts/clean.sh",shell=True)
    print("one request completed")
    return output_image, ",".join(text_output)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil"),
        gr.Textbox(label="Image Info"),
    ],
    examples=[
        ["../examples/JX_245_2_157.jpg"],
        ["../examples/YB_24_204.jpg"],
        ["../examples/06-V008P0195.jpg"],
    ],
    title="OCR system for Chinese Historical Documents(Machine Intelligence Group, MIG)"
).queue(concurrency_count=1)


# 啟動 Gradio 介面
iface.launch(share=False,server_port=9999,server_name="0.0.0.0", debug=True)
