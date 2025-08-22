# 🩺 Diabetic Retinopathy Classifier (ViT + ResNet)


This project trains and deploys deep learning models (ViT & ResNet-50) for APTOS Diabetic Retinopathy classification.  
The diagram below shows the full pipeline:

```mermaid
flowchart TB
  %% =========================
  %% OVERALL PIPELINE (DATA → TRAIN → EXPORT → EVAL → APP)
  %% =========================
  A[Start] --> B[Mount/Load Data<br/>fundus images + labels]
  B --> C[Clean & Audit Images<br/>readability checks]
  C --> D[Preprocess Fundus<br/>circle-crop + CLAHE + resize]
  D --> E[Stratified Split<br/>Train 80% / Valid 20%]
  E --> F[Compute Class Weights<br/>inverse frequency]

  %% =========================
  %% VIT TRAINING (HF / PyTorch)
  %% =========================
  subgraph G[ViT Training (PyTorch / Transformers)]
    direction TB
    G1[AutoImageProcessor (ViT-Base-384)] --> G2[Dataset (Albumentations)<br/>Train: crop/flip/rotate/brightness<br/>Valid: resize only]
    G2 --> G3[Phase 1: Head-only<br/>Freeze backbone, LR=5e-4, WD=0.05, 5–10 epochs]
    G3 --> G4[Pick best by QWK<br/>dr-vit-phase1]
    G4 --> G5[Phase 2a: Unfreeze 6–11,<br/>freeze 0–5; LR=2e-5, WD=0.10,<br/>cosine, warmup=0.10, ~20 epochs]
    G5 --> G6[Top-off (cool): LR=1e-5,<br/>warmup=0.20, early stop]
    G6 --> G7[Extended Top-off<br/>patience↑; select highest QWK]
    G7 --> G8[Export Final (HF format)<br/>dr-vit-EXPORT-final<br/>model.safetensors + config + preprocessor + classes.json]
  end

  %% =========================
  %% RESNET TRAINING (TF / Keras)
  %% =========================
  subgraph H[ResNet-50 Training (TensorFlow / Keras)]
    direction TB
    H1[tf.data / image_dataset_from_directory] --> H2[Model: ResNet-50<br/>top dense (5)]
    H2 --> H3[Train: base frozen →<br/>unfreeze = fine-tune]
    H3 --> H4[Save Keras model<br/>retinopathy_baseline_model.keras]
    H4 --> H5[NOTE: Training order is alphabetical:<br/>['Mild','Moderate','No_DR','Proliferate_DR','Severe']]
  end

  %% =========================
  %% EVALUATION (VALID & TEST)
  %% =========================
  subgraph I[Evaluation & Metrics]
    direction TB
    I1[VALID loop (HF Trainer)] --> I2[Metrics: Accuracy, F1-macro,<br/>QWK (Quadratic Weighted Kappa), AUC-macro]
    I2 --> I3[Pick best-QWK ckpt<br/>trainer_state.json.best_model_checkpoint]
    I3 --> I4[TEST loop (custom)<br/>softmax probs & predictions]
    I4 --> I5[Confusion Matrix, per-class F1,<br/>Example panels]
  end

  %% =========================
  %% LABEL ORDER / REMAP
  %% =========================
  subgraph J[Canonical Labels & Remapping]
    direction TB
    J1[APTOS canonical order:<br/>[No_DR, Mild, Moderate, Severe, Proliferative_DR]]
    J2[Model training order (alphabetical):<br/>[Mild, Moderate, No_DR, Proliferate_DR, Severe]]
    J3[Index remap (model → APTOS):<br/>{0→1, 1→2, 2→0, 3→4, 4→3}]
    J1 --> J3
    J2 --> J3
  end

  %% =========================
  %% GRADIO APP (HF Space)
  %% =========================
  subgraph K[Hugging Face Space (Gradio UI)]
    direction TB
    K1[Load ViT from dr-vit-EXPORT-final<br/>AutoImageProcessor + AutoModel]
    K2[Load ResNet .keras]
    K3[User upload: fundus image]
    K4[Predict with chosen model<br/>→ softmax probs]
    K5[Apply index remap to APTOS order]
    K6[Show top-5 probs (Label component)]
    K1 --> K4
    K2 --> K4
    K3 --> K4 --> K5 --> K6
  end

  %% WIRING BETWEEN BLOCKS
  F --> G1
  F --> H1
  G8 --> I1
  H4 --> I4
  I4 --> J1
  I4 --> J2
  J3 --> K5

  %% NOTES
  classDef note fill:#f7f7ff,stroke:#8a8aff,color:#202040
  N1:::note
  N1[QWK matters: better than accuracy for<br/>ordinal classes, penalizes distant mistakes more.]
  I2 --> N1
  ```

This project provides a web-based Gradio app to classify **Diabetic Retinopathy** severity using:
- **Vision Transformer (ViT, PyTorch)**  
- **ResNet-50 (TensorFlow/Keras)**  

The app lets you upload a fundus retinal scan and get predictions for 5 DR stages:
**No_DR, Mild, Moderate, Severe, Proliferative_DR**  
Outputs are mapped to the **APTOS clinical order** for consistency.

---

## 🚀 Demo

👉 Try the live app here:  
🔗 [Hugging Face Space](https://huggingface.co/spaces/Yeshwanth2410/DiabeticRetinopath-ViT-Resnet)

---

## 📂 Repository Structure

├── app.py                   # Gradio app: loads ViT + ResNet models
├── requirements.txt         # Dependencies
├── LICENSE                  # MIT License
└── README.md                # This file




⚠️ Note: The model weights (`.keras`, `.safetensors`) are **not stored in GitHub** because of file size limits.  
They are hosted on Hugging Face Hub instead (see below).

---

## 📦 Models

- **ViT model (final export)**:  
  Available inside the [Hugging Face Space files](https://huggingface.co/spaces/Yeshwanth2410/DiabeticRetinopath-ViT-Resnet/tree/main/dr-vit-EXPORT-final)  
  Includes:
  - `config.json`  
  - `model.safetensors`  
  - `preprocessor_config.json`  
  - `classes.json`

- **ResNet-50 model** (`retinopathy_baseline_model.keras`):  
  Also too large for GitHub. You can upload to Hugging Face Hub or provide a Google Drive link.

---

## 🔧 Installation & Local Run

Clone this repo:
```bash
git clone https://github.com/Yeshwanth-git-tech/DiabeticRetinopathy-ViT-Resnet.git
cd DiabeticRetinopathy-ViT-Resnet


## Install dependencies:

pip install -r requirements.txt

Training Notes
	•	ViT was fine-tuned in 2 phases (head-only + backbone unfreeze) with class-balanced loss.
	•	ResNet-50 was trained in Keras with a similar label remapping to match APTOS dataset order.
	•	Both models output predictions remapped into canonical order:
No_DR → Mild → Moderate → Severe → Proliferative_DR.

📜 License

This repo is licensed under the MIT License.
You are free to use, modify, and distribute it, but attribution is appreciated.





✨ Acknowledgements
	•	APTOS 2019 Blindness Detection Dataset
	•	Hugging Face Transformers
	•	Gradio