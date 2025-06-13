# This README would be modified in the future
# OCR System for Chinese Historical Documents with Image-Based Reading Order Detection

**Author:** Hsing-Yuan Ma
**Affiliation:** National Chengchi University
**Contact:** [hsingyuanma@gmail.com](mailto:hsingyuanma@gmail.com)

---

## 🚀 Project Overview

This project provides an advanced Optical Character Recognition (OCR) system tailored for **Chinese historical documents**, which are notoriously difficult for standard OCR due to their complex layouts and unique reading orders. Our system not only detects and recognizes text, but also **infers the correct reading order** using raw image features—an innovation that mirrors human reading perception.

---

## 🏆 Key Features

* **State-of-the-art text detection** with Differential Binarization++ (DB++).
* **High-accuracy text recognition** using SVTR Net.
* **Novel Reading Order Detection (ROD)** module, leveraging both visual cues and spatial information to recover authentic reading sequences.
* **Visualized output**: overlayed bounding boxes, recognized text, and reading order markers on original images.
* **Modular, cloud-deployable architecture** with a modern web interface (Gradio).

---

## 📚 System Pipeline

1. **Input**
   Upload an image of a historical Chinese document.

2. **Text Detection**
   The system locates and segments text regions using a trained DB++ model.

3. **Text Recognition**
   Each detected region is recognized as text using SVTR Net.

4. **Reading Order Detection**
   A CNN-based, multimodal model predicts the correct reading order by analyzing both image features and the spatial layout of text regions, following the "First Decide then Decode" (FDTD) algorithm.

5. **Output**

   * **Structured transcription**: digital text, ordered as in the original document.
   * **Visual output**: input image annotated with bounding boxes, text, and reading order indicators.

---

## 🧑‍🔬 Model Training

* **Dataset:** MTHv2, combining Tripitaka Koreana in Han (TKH) and Multiple Tripitaka in Han (MTH) datasets (\~3,199 images).
* **Test split:**

  * Text detection/recognition: 10%
  * Reading order detection: 30%
* **Performance:**

  * **Text Detection:** F1 score of 0.95
  * **Text Recognition:** Accuracy 0.83
  * **Reading Order Detection:** Page error rate 5%

---

## 🛠️ Getting Started

### **Requirements**

* Python 3.8+
* CUDA-enabled GPU recommended
* PaddleOCR and dependencies (see `requirements.txt`)
* Model weights (see below)

### **Installation**

1. **Clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download models**
   *(Models are not stored in this repo due to size. Run the provided download script or follow manual instructions in `README.md`.)*

   ```bash
   bash scripts/download_models.sh
   ```

   Place your detection, recognition, and ROD models in the `models/` directory.

### **Running the App**

```bash
python app/main.py
```

* The Gradio web interface will launch.
* Upload an image and receive both structured text and visualized results.

---

## 🏗️ Project Structure

```
.
├── app/                 # Core Python code (main.py, pipeline, utils, etc.)
├── models/              # Downloaded model weights (not tracked in git)
├── scripts/             # Helper scripts (e.g., download_models.sh, clean.sh)
├── output/              # Generated outputs and temp files
├── PaddleOCR/           # External OCR library
├── VORO/                # External order detection code
├── examples/            # Example images
├── requirements.txt
└── README.md
```

---

## 📝 References

* **Paper:**
  *Reading between the Lines: Image-Based Order Detection in OCR for Chinese Historical Documents*
  Hsing-Yuan Ma, Hen-Hsen Huang, Chao-Lin Liu
  AAAI-24 ([PDF link if open](https://aaai.org/ojs/index.php/AAAI/article/view/27365))

* **Citations:**

  * Liao et al. 2022: Differential Binarization++
  * Du et al. 2022: SVTR Net
  * Quiros & Vidal 2022: FDTD algorithm

---

## 🙏 Acknowledgements

This work is based on research supported by National Chengchi University and Academia Sinica.
We thank the open-source OCR and digital humanities community.

---

## 📢 License

Copyright © 2024
Association for the Advancement of Artificial Intelligence (AAAI)
Open-source for academic and research use.

---

## ✨ Contact

For questions, collaboration, or deployment advice, please contact **[hsingyuanma@gmail.com](mailto:hsingyuanma@gmail.com)**.

---

*This project helps preserve the heritage of Chinese manuscripts by making them accessible and searchable for scholars and the public.*

---

**Feel free to ask for a more concise/longer README or any specific section (cloud deploy, API usage, etc.)!**

