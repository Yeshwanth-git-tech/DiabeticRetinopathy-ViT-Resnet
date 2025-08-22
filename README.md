# 🩺 Diabetic Retinopathy Classifier (ViT + ResNet)


This project trains and deploys deep learning models (ViT & ResNet-50) for APTOS Diabetic Retinopathy classification.  
The diagram below shows the full pipeline:

```mermaid
flowchart TB
  A[Start] --> B[Load data]
  B --> C[Audit images]
  C --> D[Preprocess fundus\ncircle crop + CLAHE + resize]
  D --> E[Stratified split 80/20]
  E --> F[Class weights]

  subgraph VIT_Training
    direction TB
    G1[AutoImageProcessor] --> G2[Dataset with augmentations]
    G2 --> G3[Phase 1: head only\nLR 5e-4, WD 0.05, 5-10 epochs]
    G3 --> G4[Pick best by QWK]
    G4 --> G5[Phase 2a: unfreeze 6-11\nLR 2e-5, WD 0.10, cosine, warmup 0.10]
    G5 --> G6[Top-off cool: LR 1e-5]
    G6 --> G7[Extended top-off]
    G7 --> G8[Export: dr-vit-EXPORT-final]
  end

  subgraph ResNet_Training
    direction TB
    H1[tf.data pipeline] --> H2[ResNet50 + Dense(5)]
    H2 --> H3[Fine-tune base]
    H3 --> H4[Save: retinopathy_baseline_model.keras]
    H4 --> H5[Train order:\nMild, Moderate, No_DR, Proliferate_DR, Severe]
  end

  subgraph Evaluation
    direction TB
    I1[VALID metrics:\nAccuracy, F1 macro, QWK, AUC macro]
    I2[Select best QWK checkpoint]
    I3[TEST inference]
    I4[Confusion matrix,\nper-class F1, examples]
  end

  subgraph Label_Orders
    direction TB
    J1[APTOS order:\nNo_DR, Mild, Moderate, Severe, Proliferative_DR]
    J2[Model order:\nMild, Moderate, No_DR, Proliferate_DR, Severe]
    J3[Index remap:\n0->1, 1->2, 2->0, 3->4, 4->3]
  end

  subgraph HF_Space_Gradio
    direction TB
    K1[Load ViT export]
    K2[Load ResNet .keras]
    K3[User uploads image]
    K4[Predict -> softmax]
    K5[Apply remap -> APTOS order]
    K6[Show top-5]
  end

  F --> G1
  F --> H1
  G8 --> I1
  H4 --> I3
  I3 --> J1
  I3 --> J2
  J3 --> K5
  K1 --> K4
  K2 --> K4
  K3 --> K4 --> K5 --> K6
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