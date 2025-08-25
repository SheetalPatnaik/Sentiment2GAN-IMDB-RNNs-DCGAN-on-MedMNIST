# 🎬 Sentiment2GAN — Sentiment Analysis (IMDB) + DCGAN on MedMNIST

This project brings together **Natural Language Processing (NLP)** and **Generative Modeling (GANs)** to demonstrate how Deep Learning can be applied across very different domains — text (sentiment analysis) and medical imaging (synthetic generation).  
Developed as part of **Deep Learning coursework (DATA 255 @ SJSU)**, it highlights my ability to build **end-to-end pipelines, compare architectures, and evaluate models with both qualitative and quantitative metrics**.

---

## 🔑 Project Highlights
### Part 1 — IMDB Sentiment Analysis
- **Dataset:** IMDB Movie Reviews (50k reviews)  
- **Text Preprocessing:** tokenization, lowercasing, stopword removal, HTML stripping, lemmatization/stemming.  
- **Word Embeddings:** Word2Vec for feature representation.  
- **Architectures compared:**
  - RNN  
  - LSTM  
  - GRU  
  - BiLSTM  
- **Training setup:** 50+ iterations with early stopping, train/val/test split.  
- **Results:** BiLSTM achieved the highest accuracy, demonstrating strong sequential context capture.  

---

### Part 2 — DCGAN on MedMNIST
- **Dataset:** MedMNIST (PathMNIST / BloodMNIST etc.)  
- **Generator architecture:** deconvolution-based, trained for ≥1000 epochs.  
- **Discriminator:** 3+ convolutional layers with BatchNorm + LeakyReLU.  
- **Evaluation:**
  - Generator & Discriminator loss curves for training dynamics.  
  - **FID Score** (1000 real vs 1000 generated samples) for quantitative evaluation.  
  - Mode collapse detection using both loss curves and generated samples.  
- **Results:** Generated 32+ synthetic medical images; evaluated quality vs mode collapse risk.  

---

## 📊 Key Results
- **Sentiment Analysis:**  
  - RNN/GRU performed well but LSTM and BiLSTM achieved **best accuracy and generalization**.  
- **DCGAN:**  
  - Generated realistic MedMNIST images after 1000 epochs.  
  - **FID scores confirmed diversity**, though occasional mode collapse was observed.  
  - Static evaluation of samples showed reasonable class variability.  

---

## ⚙️ Tech Stack
- **Frameworks:** PyTorch, Keras, TensorFlow  
- **NLP:** Word2Vec, RNN, LSTM, GRU, BiLSTM  
- **GANs:** DCGAN (Generator + Discriminator, PyTorch implementation)  
- **Metrics:** Accuracy (NLP), FID Score (GANs), Loss curves (training stability)  


---

## ▶️ Quickstart
```bash
# 1. create environment
python -m venv .venv
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run notebooks
jupyter lab

