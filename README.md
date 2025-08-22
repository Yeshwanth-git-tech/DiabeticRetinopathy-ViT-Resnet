# 🩺 Diabetic Retinopathy Classifier (ViT + ResNet)


This project trains and deploys deep learning models (ViT & ResNet-50) for APTOS Diabetic Retinopathy classification.  
The diagram below shows the full pipeline:

```mermaid
 flowchart TB
     %% --- Style Definitions ---
     classDef bronzeStyle fill:#cd7f32,stroke:#8B4513,stroke-width:2px,color:#ffffff
     classDef silverStyle fill:#bdc3c7,stroke:#7f8c8d,stroke-width:2px,color:#000000
     classDef goldStyle fill:#f1c40f,stroke:#b8860b,stroke-width:2px,color:#000000
 
     %% --- Bronze Tier ---
     subgraph Bronze Tier - Data Foundation
         direction TB
         A["📄 Load APTOS 2019 Dataset"] --> B["🔍 Audit & Clean Images"]
         B --> C["✨ Advanced Preprocessing<br/>(Fundus Crop + CLAHE)"]
         C --> D["⚖️ Stratified 80/20 Split"]
         D --> E["🎚️ Compute Class Weights"]
     end
     class A,B,C,D,E bronzeStyle
 
     %% --- Silver Tier ---
     subgraph Silver Tier - Model Training
         direction LR
 
         subgraph "ViT Training (PyTorch)"
             direction TB
             G1["ViT AutoImageProcessor"] --> G2["Custom PyTorch Dataset<br/>(Albumentations Augmentation)"]
             G2 --> G3["<b>Phase 1:</b> Head Training<br/><i>(5 Epochs, LR: 5e-4)</i>"]
             G3 --> G4["<b>Phase 2:</b> Fine-Tune<br/><i>(20 Epochs, LR: 2e-5)</i>"]
             G4 --> G5["📦 Export ViT Model"]
         end
 
         subgraph "ResNet-50 Training (TensorFlow)"
             direction TB
             H1["tf.data Pipeline"] --> H2["<b>Model:</b> ResNet50 Base + Dense(5)<br/><i>(23.5M Params)</i>"]
             H2 --> H3["<b>Initial Training:</b> Train Head<br/><i>(25 Epochs, LR: 1e-3)</i>"]
             H3 --> H4["<b>Fine-Tuning:</b> Unfreeze Base<br/><i>(10 Epochs, LR: 1e-5)</i>"]
             H4 --> H5["💾 Save ResNet-50 Model"]
         end
     end
     class G1,G2,G3,G4,G5,H1,H2,H3,H4 silverStyle
 
     %% --- Gold Tier ---
     subgraph Gold Tier - Prediction
         direction LR
         I1["📤 Upload Retinal Scan"] --> I2["🤖 Model Prediction"]
         I2 --> I3["🩺 Predicts: No_DR"]
         I2 --> I4["🩺 Predicts: Mild"]
         I2 --> I5["🩺 Predicts: Moderate"]
         I2 --> I6["🩺 Predicts: Severe"]
         I2 --> I7["🩺 Predicts: Proliferate_DR"]
     end
     class I1,I2,I3,I4,I5,I6,I7 goldStyle
 
     %% --- Connections ---
     E --> G1
     E --> H1
     G5 --> I1
     H5 --> I1
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