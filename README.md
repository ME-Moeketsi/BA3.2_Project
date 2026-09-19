# Predict-Care AI

### AI-Driven Patient Management & Healthcare Operations Decision Support System

Predict-Care AI is an AI-driven healthcare system designed to support patient management, triage analysis, patient flow, and hospital operational decision-making.

The system uses machine learning to analyse patient and hospital operational data, identify patterns, and provide predictions that can assist healthcare staff in making informed operational decisions.

## Project Objectives

* Support AI-assisted patient triage analysis.
* Improve understanding of patient flow and hospital workload.
* Predict hospital waiting times and operational outcomes.
* Analyse peak-time hospital conditions.
* Support healthcare resource planning and allocation.
* Provide data-driven decision support for healthcare staff.

## AI & Machine Learning

The project currently focuses on two main machine-learning components:

### 1. Patient Triage Prediction

Uses emergency patient data to analyse patient characteristics and predict the associated triage category.

**Algorithm:** Random Forest Classifier

**Target:** `KTAS_expert`

### 2. Hospital Performance Prediction

Uses hospital simulation data to analyse operational conditions and predict hospital performance outcomes such as waiting times.

**Algorithm:** Random Forest Regressor

**Key operational factors:**

* Patient priority
* Peak arrival conditions
* Number of doctors
* Peak limits
* Hospital scenario

**Key performance outcomes:**

* Doctor waiting time
* Medication waiting time
* Total time in hospital

## Datasets

### Emergency Service Triage Dataset

Used for patient and triage analysis.

The dataset contains emergency-department patient information including demographic information, arrival details, pain, vital signs, injury, mental state, and triage classifications.

### Hospital Simulation Dataset

Used for analysing hospital operations under different scenarios, including peak arrival conditions, staffing levels, patient priority, service times, and waiting times.

## System Concept

```text
Patient Information
        ↓
AI Triage Analysis
        ↓
Triage Prediction
        ↓
Patient Flow
        ↓
Hospital Operational Analysis
        ↓
Waiting-Time / Performance Prediction
        ↓
AI Decision Support
        ↓
Healthcare Staff
```

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Statsmodels
* Joblib
* Machine Learning
* Time-Series Analysis

## Project Status

**Under Development**

The current development stage focuses on preparing the datasets, developing the machine-learning models, evaluating predictions, and integrating the AI/ML component into the wider Predict-Care AI system.

## Scope

Predict-Care AI is intended as a **decision-support system**. It is designed to assist healthcare staff with data analysis, predictions, patient-flow understanding, and operational planning.

It does not replace healthcare professionals or make autonomous clinical decisions.

---

**Predict-Care AI**
*Using AI to support smarter healthcare operations and patient management.*
