# Oral Lesion Detection AI
### Navigation Map
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [About the Dataset](#about-the-dataset)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Acknowledgements](#acknowledgements)

---

### Overview
This project implements a deep learning-based computer vision system to automatically detect early-stage oral abnormalities from clinical mouth images. The goal is binary classification: determining whether an image shows a healthy oral cavity or signs of a potentially harmful lesion.

---

### Environment Setup
For the portal to run, python packages will be required .This whole portal is build on python3.11 so the required packages are mentioned in a file named as requirements.txt from where we can download all the required libraries in a single go by just typing the following command in the command prompt after installing python.
```bash
pip install -r requirements.txt
```
---

### About the Dataset
This project utilizes the Oral Cancer Image Dataset from Kaggle, which contains a collection of labeled clinical images representing both healthy oral cavities and those with visible lesions. The dataset is well-balanced, making it suitable for training deep learning models for binary classification.

Curated to support research in automated oral lesion detection, the dataset provides high-quality images that facilitate the development of accurate and reliable diagnostic tools. It serves as a valuable resource for advancing early detection and intervention techniques in oral healthcare through machine learning and computer vision.

---

### Project Structure
```text
oral-lesion-detection-ai/
├── artifacts/
├── config/
│   └── config.yaml
├── logs/
│   └── running_logs.log
├── research/
│   └── notebooks/
├── src/
│   ├── cnnClassifier/
│   │   ├── components/
│   │   ├── config/
│   │   ├── constants/
│   │   ├── entity/
│   │   ├── pipeline/
│   │   └── utils/
│   └── __init__.py
├── templates/
│   └── index.html
├── .gitignore
├── dvc.yaml
├── main.py
├── params.yaml
├── README.md
├── requirements.txt
├── scores.json
├── setup.py
└── template.py
```

---

### Development Workflow
1. Update config.yaml – Define file paths, model configurations, and directory structure
2. Update params.yaml – Set model training parameters such as image size, learning rate, batch size, etc.
3. Update Entity Classes – Define structured data models using Python dataclasses
4. Update Configuration Manager (src/cnnClassifier/config) – Read and validate configurations
5. Update Components – Implement core logic like data ingestion, model creation, training, and evaluation
6. Update Pipeline Stages – Connect components to form modular pipeline stages
7. Update main.py – Run all pipeline stages sequentially using a central controller
8. Update dvc.yaml – Define and version the DVC pipeline for reproducibility
9. Update or Run app.py – Launch the Flask web application for image-based predictions

---

### Acknowledgements
I would like to thank the individuals, tools, and open-source communities that contributed—directly or indirectly—to the development of this project:
- ZaidPy – Oral Cancer Image Dataset – For providing a high-quality dataset for training and evaluation
- TensorFlow & Keras – For enabling efficient deep learning model development
- Flask – For supporting quick and easy model deployment as a web application
- DVC and MLflow – For experiment tracking, data versioning, and pipeline reproducibility
- DagsHub – For integrating version control, data tracking, and ML experiment management in one collaborative platform
- VS Code, Git & GitHub – For providing a stable development environment and version control workflow
- Official Documentation & Developer Guides – Referred throughout the project for TensorFlow, Flask, DVC, MLflow, and other tools to ensure best practices and correct implementation
- Open Source Community – For tutorials, blog posts, and community support that greatly enhanced my understanding and helped troubleshoot challenges
This project is a reflection of collective knowledge shared by the global developer community—thank you!