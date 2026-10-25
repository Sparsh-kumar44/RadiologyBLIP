# RadiologyBLIP 🩻

### Medical Visual Question Answering using BLIP

RadiologyBLIP is a deep learning project that explores **Medical Visual Question Answering (VQA)** by fine-tuning a pretrained BLIP (Bootstrapping Language-Image Pre-training) model on the VQA-RAD dataset.

The model takes a radiology image and a natural-language question as input and generates a text-based answer. The project investigates how a pretrained vision-language model can be adapted to the medical imaging domain.

## 🎯 Objectives

- Fine-tune a pretrained BLIP model for medical visual question answering.
- Process radiology images and their corresponding questions and answers.
- Implement a custom PyTorch dataset and data-loading pipeline.
- Apply gradient accumulation to support training with limited GPU memory.
- Compare the pretrained model's predictions with those of the fine-tuned model.
- Save the fine-tuned model for future inference.

## 🧠 Model and Dataset

### BLIP

BLIP is a vision-language model designed to work with both images and text. In this project, the pretrained BLIP model is adapted to answer questions about radiology images.

### VQA-RAD

The project uses the **VQA-RAD (Visual Question Answering in Radiology)** dataset, which contains medical images paired with natural-language questions and answers.

Each training example consists of:

- **Image:** A radiology image.
- **Question:** A natural-language question about the image.
- **Answer:** The corresponding target answer.

## ⚙️ Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Model and Processor:** Hugging Face Transformers
- **Data Processing:** Pandas, NumPy
- **Image Processing:** Pillow
- **Progress Tracking:** tqdm
- **Environment:** Jupyter Notebook / Google Colab

## 🔄 Workflow

1. **Data preprocessing:** Load the VQA-RAD annotations, remove unnecessary fields and invalid records, and prepare the image-question-answer pairs.
2. **Dataset preparation:** Implement a custom PyTorch `Dataset` and `DataLoader` to process images, questions, and target answers.
3. **Baseline inference:** Generate predictions using the pretrained BLIP model before fine-tuning.
4. **Model fine-tuning:** Train BLIP on the prepared dataset using the AdamW optimizer and supervised learning.
5. **Gradient accumulation:** Accumulate gradients across multiple mini-batches before updating model parameters to help manage GPU memory usage.
6. **Loss calculation:** Mask padding tokens in the target labels so that they are ignored during loss calculation.
7. **Evaluation:** Compare the pretrained and fine-tuned models on test examples.
8. **Model saving:** Save the fine-tuned model and processor for later use.

## 📁 Repository Structure

```text
RadiologyBLIP/
├── blip_vqa_finetuned/   # Saved fine-tuned model (ignored by Git)
├── images/               # Radiology image files
├── RadiologyBLIP.ipynb   # Main notebook
├── vqa_rad.json          # Dataset annotations
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation
```

The `blip_vqa_finetuned/` directory is excluded from version control because model checkpoints can be large. The directory may exist locally after training without being included in the GitHub repository.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd RadiologyBLIP
```

Replace `<YOUR_REPOSITORY_URL>` with your GitHub repository URL.

### 2. Create a Python environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install torch transformers pandas numpy pillow tqdm scikit-learn jupyter
```

For GPU training, install a PyTorch version compatible with your CUDA environment using the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Prepare the dataset

Ensure that:

- `vqa_rad.json` is available at the path expected by the notebook.
- The `images/` directory contains the corresponding radiology images.
- Image filenames in the annotation file match the actual filenames.

The dataset and image paths may need to be adjusted depending on whether you run the notebook locally or in Google Colab.

### 5. Run the notebook

Open `RadiologyBLIP.ipynb` in Jupyter Notebook or upload it to Google Colab.

Run the notebook cells in order to perform data preprocessing, baseline inference, model fine-tuning, and evaluation.

## 💾 Model Checkpoints

After fine-tuning, the model and processor are saved in the `blip_vqa_finetuned/` directory.

This directory can be used for subsequent inference without repeating the training process, provided the saved model files are available.

Because the checkpoint directory is excluded by `.gitignore`, it will not automatically be uploaded to GitHub. If you want to share the trained model, consider publishing it to the [Hugging Face Hub](https://huggingface.co/docs/hub/index).

## 📊 Evaluation

The pretrained and fine-tuned models can be evaluated using the same test examples to compare their generated answers.

Possible evaluation methods include:

- **Exact-match accuracy:** Measures the proportion of predictions that exactly match the reference answers.
- **Qualitative analysis:** Inspects generated answers for correctness and relevance.
- **Question-type analysis:** Examines model performance across different types of medical questions.

Actual evaluation metrics should be reported here after running the final evaluation. Fine-tuning should not be assumed to improve performance unless the results demonstrate it.

## 🔮 Future Improvements

- Train for additional epochs and monitor validation loss.
- Experiment with learning rates, batch sizes, and gradient accumulation steps.
- Evaluate with appropriate medical VQA metrics.
- Analyze incorrect predictions and common failure cases.
- Compare BLIP with alternative vision-language models.


