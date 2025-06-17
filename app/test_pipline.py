import logging
import os

from config import FILE_DICT
from main import process_image
from PIL import Image
from pipline import clean_temp


def dummy_progress(p, desc=""):
    logging.info(f"{desc} ({int(p*100)}%)")


def test_process_image():
    # Use an image from your examples
    example_image_path = "./examples/JX_245_2_157.jpg"
    assert os.path.isfile(
        example_image_path
    ), f"Test image missing: {example_image_path}"

    # Choose a unique test filename
    test_filename = "test_image.jpg"

    # Load image
    img = Image.open(example_image_path)

    # Call the pipeline
    try:
        img_vis, seq_txt = process_image(test_filename, img, dummy_progress)
        logging.info("Image and text sequence returned successfully.")
    except Exception as e:
        logging.error("Pipeline crashed:", e)
        assert False, f"Pipeline failed with exception: {e}"

    # Check outputs
    processed_img_path = os.path.join(FILE_DICT["PROCESSED_FOLDER"], test_filename)
    assert os.path.exists(
        processed_img_path
    ), f"Processed image not found: {processed_img_path}"
    assert img_vis is not None, "No visualization image returned."
    assert seq_txt is not None, "No sequence text returned."
    # run clean bash
    clean_temp(debug=True)
    logging.info("All checks passed!")


if __name__ == "__main__":
    test_process_image()
