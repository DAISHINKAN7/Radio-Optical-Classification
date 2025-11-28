<p align="center">
  <img src="assets/banner.png" width="92%"/>
</p>

<h1 align="center">🌌 Radio–Optical Galaxy Morphology Classification</h1>
<h3 align="center">ConvNeXt · EfficientNet · ResNet · ViT · Weighted Ensemble (10-Run Statistical Evaluation)</h3>

<p align="center">
  Multimodal radio + optical cutouts across <b>six astrophysical classes</b> classified using state-of-the-art deep learning architectures<br>
  with robust 10-run evaluation, confidence intervals, and an ensemble model achieving peak performance.
</p>

---

<p align="center">

  <!-- Python -->
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" />

  <!-- PyTorch -->
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" />

  <!-- Multi-Run Eval -->
  <img src="https://img.shields.io/badge/Evaluation-10--Run-orange" />

  <!-- Ensemble -->
  <img src="https://img.shields.io/badge/Model-Ensemble%20Learning-purple" />

  <!-- Galaxy Classes -->
  <img src="https://img.shields.io/badge/Classes-6-brightgreen" />

  <!-- MIT License -->
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />

</p>

---

# 📁 Dataset Overview

This project uses **paired radio + optical cutouts** for each galaxy object.  
Each sample consists of:

- **LOFAR / Radio image**  
- **Optical (Pan-STARRS / SDSS) cutout**  
- **Image-level label** (one of 6 classes)

### **Supported Classes**
1. **Compact Sources**  
2. **Elliptical Galaxies**  
3. **FR2 Radio Galaxies**  
4. **Radio-Loud AGN**  
5. **Spiral Galaxies**  
6. **Starburst Galaxies**

All models are trained on **RGB-stacked multimodal input** (radio + optical channels).

---

# ⚙️ Project Structure

```
project/
│── data/
│   ├── train/
│   ├── val/
│   └── test/
│
│── models/
│   ├── efficientnet.py
│   ├── convnext.py
│   ├── resnet.py
│   ├── vit.py
│   └── ensemble.py
│
│── training/
│   ├── train_single.py
│   ├── train_gan.py
│   └── train_ensemble.py
│
│── evaluation/
│   ├── evaluate.py
│   ├── multi_run_eval.py
│   └── metrics.py
│
│── outputs/
│   ├── confusion_matrices/
│   ├── run_variations/
│   └── final_models/
│
└── README.md
```

---

# 🚀 Training

### **Train a single model**
```
python training/train_single.py --model convnext --device cuda
```

### **Run 10-run evaluation**
```
python evaluation/multi_run_eval.py --model efficientnet --runs 10
```

### **Train GAN (Radio → Optical Translation)**
```
python training/train_gan.py ^
    --data_path data\beautiful_dataset_v2 ^
    --output_dir outputs\gan_radio2optical ^
    --mode radio2optical ^
    --batch_size 4 ^
    --num_epochs 200 ^
    --device cuda
```

### **Weighted Ensemble Inference**
The ensemble combines **ConvNeXt + EfficientNet** using confidence-based soft voting.

```
python inference/ensemble_predict.py --input sample.png
```

---

# 🧬 Model Zoo

Below is the complete list of models implemented and evaluated in this project  
including their 10-run mean performance (Accuracy ± 95% CI):

| Model                | Architecture Type | Accuracy (95% CI)  | F1-Score  | AUC     | Notes |
|---------------------|-------------------|--------------------|-----------|---------|-------|
| **ConvNeXt-Tiny**    | Modern CNN        | **96.59% ± 0.22%** | **0.966** | 0.998   | Best single model; highly stable |
| **EfficientNet-B0**  | Scaled CNN        | 95.35% ± 0.42%     | 0.953     | 0.997   | Strong + efficient |
| **ResNet-50**        | Classical CNN     | 92.09% ± 0.60%     | 0.921     | 0.992   | Baseline benchmark |
| **ViT**              | Transformer       | 85.39% ± 0.53%     | 0.852     | 0.978   | Underperforms on this domain |
| **Ensemble (ConvNeXt + EfficientNet)** | Weighted Voting | **97.60% ± 0.31%** | **0.976** | **0.999** | Best overall; statistically superior |

---

# 📊 Evaluation Highlights

### ✔ 10-run stability plots  
### ✔ Confidence Intervals (95% CI)
### ✔ Aggregated confusion matrices  
### ✔ Per-class F1-score comparison  
### ✔ Ensemble outperforming all baselines  
### ✔ GAN-based augmentation option  

All plots and tables are stored in:

```
outputs/run_variations/
outputs/confusion_matrices/
outputs/metrics/
```

---

# 🧩 Ensemble Method

The ensemble uses:

- **ConvNeXt probability vector**
- **EfficientNet probability vector**
- **Learned or fixed weights (default 0.6 / 0.4)**

Final prediction:

```
p_final = w1 * p_convnext + w2 * p_efficientnet
```

This increased accuracy by **+0.91%** compared to the best standalone model.

---

# 🖼️ Results Summary

- ConvNeXt is the **strongest individual model**
- EfficientNet is **more parameter-efficient**
- ResNet provides a **strong baseline**
- ViT underperforms for radio-optical data
- The **ensemble achieves the best accuracy: 97.60%**
- Very small **standard deviation**, showing high stability

---

# 📦 Installation

```
git clone https://github.com/youruser/yourrepo
cd yourrepo
pip install -r requirements.txt
```

---

# 📜 License

This project is released under the **MIT License**.

---

# 🤝 Acknowledgements

- LOFAR, Pan-STARRS, SDSS  
- PyTorch  
- Scientific community in radio astronomy  
- Model architectures by FAIR, Google Brain, Microsoft Research  

---