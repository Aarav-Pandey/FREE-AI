# 🧠 FREEAI — Local Question Answering System

## 📌 Project Overview
FREEAI is a **locally hosted Question Answering (QA)** web application that allows users to ask questions based on given context paragraphs.  
It uses a **fine-tuned DistilBERT model** for question answering, trained on **SQuAD v2.0** data, and served using **Flask + Waitress** for a fast, production-ready backend.

---

## 🚀 Features
- ⚡ **Offline & Secure** — No internet or external API required.  
- 🧩 **Model Inference via Local DistilBERT**.  
- 💬 **Interactive Web UI** for entering context and asking questions.  
- 💻 **GPU Acceleration** supported (if CUDA is available).  
- 🎨 **Modern, ChatGPT-style Interface** with your own logo and branding.

---

## 🧱 Project Structure

FREEAI/
│
├── app.py                      # Flask web app (main entry point)
│
├── scripts/
│   ├── train_qa.py             # Model training and fine-tuning script
│   ├── train_json.py           # Training script for SQuAD-style JSON data
│   ├── train_csv.py            # Training script for CSV-based datasets
│   ├── simple_eval.py          # Evaluation script for testing model accuracy
│   └── local_qa.py             # Model inference (QA pipeline for local use)
│
├── web/
│   ├── index.html              # Frontend HTML layout for QA interface
│   ├── style.css               # ChatGPT-style CSS styling
│   └── script.js               # JavaScript for interactivity and API calls
│
├── models/
│   └── local_distilbert/       # Directory containing fine-tuned QA model
│
├── data/
│   ├── train-v2.0.json         # SQuAD-style training dataset
│   ├── dev-v2.0.json           # Development/validation dataset
│   └── qa_dataset.csv               # Optional CSV-based dataset
│
├── output/
│   └── predictions.json        # Model-generated predictions after evaluation
│
├── static/
│   └── logo.png                # Optional logo for the UI
│
├── requirements.txt            # Project dependencies list
│
└── README.md                   # Documentation and setup guide



Download all files and folders and place them in the same order as given in github repository to your D drive inside FREEAI folder if not create one


Requirements:
Python 3.10 or higher
pip
GPU (optional but recommended)


Nest open windows terminal Then :
1.Create a Virtual Environment:
python -m venv venv

2.Activate Virtual Environment
.\venv\Scripts\activate.ps1

3.Install Dependencies
pip install -r requirements.txt

4.Train model
pip install --upgrade pip setuptools wheel
python D:\FREEAI\scripts\train_json.py
python D:\FREEAI\scripts\train_qa.py

5.Test / Evaluate Model
python D:\FREEAI\scripts\simple_eval.py D:\FREEAI\data\dev-v2.0.json D:\FREEAI\output\predictions.json

6.Run the Local Web App
Open another terminal Then write the following commands
.\venv\Scripts\Activate.ps1
waitress-serve --listen=127.0.0.1:5000 app:app
wait for a while then use ctrl+click on http://127.0.0.1:5000 or directly open this link in your browser

7.Where to get questions and context to run in browser
Open questions.txt downloaded from github repository present in your folder
Copy paste context in context box and copy paste question in question block and then click on ask button to get answer
To clear tap on clear button then you can ask as many questions as you want

THANK YOU
