# Ocular-disease-recognization-
# 🩺 Automatic Detection of Ocular Diseases Using Deep Learning and Local Binary Patterns

## 📘 Overview

This project presents a **deep learning-based system for automated ocular disease detection**, with a focus on **binary classification of cataract vs. normal eyes**.
By combining **Local Binary Pattern (LBP)** texture features with **deep neural networks (VGG19, ResNet50, and Vision Transformer)**, the system enhances feature learning and achieves high accuracy on the **ODIR (Ocular Disease Intelligent Recognition)** dataset.

---

## 🧠 Research Motivation

Ocular diseases such as **diabetic retinopathy**, **glaucoma**, **age-related macular degeneration**, and **cataracts** are among the top causes of blindness worldwide.
Manual diagnosis from retinal fundus images is often subjective and time-consuming.
This project aims to automate early detection using **hybrid deep learning and classical feature extraction (LBP)** to improve accuracy and interpretability.

---

## 📊 Dataset

* **Dataset Used:** [ODIR - Ocular Disease Intelligent Recognition (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
* **Total Samples:** 5,000 patients (both left and right eye fundus images)
* **Classes in ODIR:** Normal (N), Diabetes (D), Glaucoma (G), Cataract (C), Hypertension (H), Pathological Myopia (M), Age-related Macular Degeneration (A), and Others (O)
* **Focus of this Study:** Binary Classification → **Cataract vs Normal**
* **Final Dataset:** 594 Cataract images + 509 Normal images
* **Split:** 80% Training, 20% Testing

---

## 🧩 Methodology

### 1. **Data Preprocessing**

* Image resizing:

  * CNNs: 224×224
  * Vision Transformer: 128×128
* Normalization to [0, 1]
* Data augmentation: random flips, rotations, zooming

### 2. **Feature Extraction using LBP**

* LBP captures **texture-based local variations** useful for identifying **lens opacity** (cataract).
* Produces histograms of binary patterns representing texture distribution.
* Improves model sensitivity to fine-grained retinal textures.

### 3. **Deep Learning Architectures**

| Model                        | Type        | Description                                                                              |
| ---------------------------- | ----------- | ---------------------------------------------------------------------------------------- |
| **VGG19**                    | CNN         | 19-layer deep CNN pre-trained on ImageNet; fine-tuned for binary classification.         |
| **ResNet50**                 | CNN         | Residual Network (50 layers) using skip connections for better gradient flow.            |
| **Vision Transformer (ViT)** | Transformer | Patch-based transformer architecture for image classification using attention mechanism. |

### 4. **Transfer Learning**

* Pretrained weights from **ImageNet** used.
* Only deeper layers fine-tuned.
* Optimizers: Adam / AdamW
* Loss: Binary Cross-Entropy (for CNNs), Sparse Categorical Cross-Entropy (for ViT).
* Early stopping and checkpointing used to prevent overfitting.

### 5. **Hybrid Approach**

Models were trained **with** and **without LBP** preprocessing to evaluate the effect of texture features.

---

## 📈 Results Summary

| Model                  | With LBP | Validation Accuracy | Validation Loss | Observation                          |
| ---------------------- | -------- | ------------------- | --------------- | ------------------------------------ |
| **VGG19**              | ❌        | 99.08%              | 0.0849          | High accuracy, slight overfitting    |
| **ResNet50**           | ❌        | 97.70%              | 0.1315          | Mild overfitting                     |
| **Vision Transformer** | ❌        | 78.16%              | 0.4764          | Underfitting observed                |
| **VGG19**              | ✅        | 94.44%              | 0.0996          | Accuracy drop, overfitting increased |
| **ResNet50**           | ✅        | **100%**            | **0.0082**      | Best generalization and performance  |
| **Vision Transformer** | ✅        | 85.71%              | 0.3453          | Improved generalization              |

**Key Findings:**

* LBP-enhanced preprocessing significantly improved **ResNet50’s** performance.
* Vision Transformer showed improvement but needs more data to generalize well.
* CNNs (especially ResNet50) outperform transformers for limited data scenarios.

---

## 🧮 Evaluation Metrics

* **Accuracy:** (TP + TN) / (TP + TN + FP + FN)
* **Precision:** TP / (TP + FP)
* **Recall:** TP / (TP + FN)
* **F1-Score:** 2 × (Precision × Recall) / (Precision + Recall)
* Confusion Matrix and Generalization Analysis performed to assess overfitting.

---

## 🔍 Visual Workflow

```
ODIR Dataset → Preprocessing → LBP (optional) → CNN / ViT Models
         ↓                           ↓
   Augmentation (flip, rotate)    Training (Transfer Learning)
         ↓                           ↓
        Evaluation → Metrics → Results Visualization
```

---

## 🧾 Conclusion

* The **hybrid deep learning approach** combining **LBP** with CNNs (especially ResNet50) demonstrated strong potential for **automated cataract detection**.
* **LBP preprocessing** improved generalization and reduced validation loss.
* Future directions:

  * Extend to **multiclass classification** (e.g., DR, Glaucoma, AMD).
  * Explore **ensemble learning** combining CNNs and Transformers.
  * Test on **cross-dataset validation** for robustness in real-world applications.

---

## 🛠️ Tools & Technologies

* **Language:** Python 3.10
* **Libraries:** TensorFlow, Keras, NumPy, OpenCV, Matplotlib, Scikit-learn
* **Models:** VGG19, ResNet50, Vision Transformer (ViT)
* **Dataset Source:** Kaggle (ODIR-5K)

---

## 📚 References

The full list of references can be found in the published research paper.
Notable works include studies on **EfficientNetB7**, **MobileNetV2/V3**, **InceptionV3**, and **Transformer-based ocular disease detection**.

---

## 👩‍💻 Authors

**Suhani Yadav**, **Aarti Rathee**, **Pranshu Kumar**
*(Published: 19 May 2025)*

---

## 🏁 Citation

If you use this repository or dataset, please cite:

> *Yadav, S., Rathee, A., & Kumar, P. (2025). Automatic Detection of Ocular Diseases Using Deep Learning and Local Binary Patterns.*

---

## 📂 Repository Structure

```
Ocular-Disease-Recognition/
│
├── data/
│   └── ODIR_dataset/
│
├── models/
│   ├── vgg19_model.h5
│   ├── resnet50_model.h5
│   └── vit_model.h5
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── lbp_feature_extraction.ipynb
│   └── model_training.ipynb
│
├── results/
│   ├── accuracy_curves/
│   └── confusion_matrices/
│
├── README.md
└── requirements.txt
```

---

## 📎 License

This project is licensed under the **MIT License** — feel free to use and modify with proper attribution.
