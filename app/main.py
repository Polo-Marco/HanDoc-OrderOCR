import logging
import os
from datetime import datetime

import gradio as gr
from config import DEBUG, FILE_DICT, RESULT_FILES
from pipline import clean_temp, detect_order, detect_text, recognize_text
from utils import vis_det_rec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def process_image(filename, input_image, progress):
    logging.info(f"Processing {filename}")
    ori_img_path = os.path.join(FILE_DICT["SAVE_FOLDER"], filename)
    # save image first
    progress(0.1, desc="Saving image")
    input_image.save(ori_img_path)
    progress(0.2, desc="Detecting texts")
    detect_text(debug=DEBUG)
    progress(0.5, desc="Detecting order")
    detect_order(ori_img_path, debug=DEBUG)
    progress(0.7, desc="Recognizing texts")
    recognize_text(ori_img_path, debug=DEBUG)
    progress(0.8, desc="Visualizing results")
    # visualize det, rec, order results
    img_vis, seq_txt = vis_det_rec(
        ori_img_path,
        RESULT_FILES["DET_RESULT"],
        RESULT_FILES["REC_RESULT"],
        RESULT_FILES["ORDER_RESULT"],
    )
    img_vis.save(os.path.join(FILE_DICT["PROCESSED_FOLDER"], filename))
    return img_vis, seq_txt


# set examples
example_folder = "./examples"
examples = [
    [os.path.join(example_folder, fname)]
    for fname in sorted(os.listdir(example_folder))
    if os.path.isfile(os.path.join(example_folder, fname))
]


def gradio_interface(input_image, progress=gr.Progress()):
    try:
        # before process clean temp
        clean_temp(DEBUG)
        # human readable input naming
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        output_image, text_output = process_image(filename, input_image, progress)
        # run clean bash
        clean_temp(DEBUG)
        logging.info("One request completed")
        return output_image, ",".join(text_output)
    except Exception as e:
        logging.error(f"Error in Gradio interface: {e}")
        return None, f"Error: {str(e)}"


# gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil"),
        gr.Textbox(label="Image Text"),
    ],
    examples=examples,
    title="OCR system for Chinese Historical Documents(Machine Intelligence Group, MIG)",
).queue(concurrency_count=1)

if __name__ == "__main__":
    # launch gradio
    iface.launch(share=False, server_port=9999, server_name="0.0.0.0", debug=DEBUG)
