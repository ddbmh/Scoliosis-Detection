# Scoliosis Detection 

This repository contains the code and experimental history for final project. The objective of this project is to detect and classify scoliosis from X-ray images using Deep Learning techniques.

We established a strong baseline using ResNet and progressively improved performance through various experiments involving loss functions, fine-tuning strategies, and model architecture variations. The final version's model type  is DenseNet201.
## 📂 File Structure & Project Evolution

The codebase reflects the iterative development process of the project, moving from a simple baseline to a more sophisticated, interpretable model.

| Filename | Description |
| :--- | :--- |
| **`baseline_model.py`** | **The Baseline:** The initial implementation. It sets up the data loader and trains a standard **ResNet** backbone to establish a performance baseline. |
| **`baseline_model_with_seed_and_gradcam.py`** | **Reproducibility & Explainability:** It ensures reproducibility by fixing random seeds and integrates **Grad-CAM** to visualize the regions of the spine the model focuses on. |
| **`baseline_model_with_finetune.py`** | **Transfer Learning:** Implements fine-tuning strategies. This script explores freezing different layers fine-tuning to adapt pre-trained weights to our medical dataset. |
| **`baseline_model_with_focalloss.py`** | **Handling Imbalance:** Addresses class imbalance issues in the dataset by replacing standard Cross-Entropy Loss with **Focal Loss**, focusing the model on hard-to-classify examples. |
| **`baseline_model_with modeltype.py`** | **The final version:** The default model is DenseNet201.  An extension of the baseline that allows for modular switching between different model architectures (e.g., ResNet-50, DenseNet201) to compare their effectiveness. |

```bash
python baseline_model_*.py
