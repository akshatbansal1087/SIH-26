# SIH 2026 Project

## 1. Project Information

- **Project Title:** N.E.T.R.A. – Neural Engine for Tropical Recognition and Analysis
- **PS ID:** SIH26070
- **PS Title:**  To develop an AI/ML based system for identification, classification, and prediction of different tropical cyclone patterns using multi-source satellite data.
- **Category:** Software
- **Theme:** Disaster Management

## 2. Problem Statement

Forecasting cyclones manually is too slow and risky, so we need an AI system that instantly crunches satellite data to spot, categorize, and track these storms before they strike.

## 3. Proposed Solution

N.E.T.R.A. allows a user to upload paired Brightness Temperature (BT) and RAW satellite images. The backend processes the images using a two-stream machine-learning model, predicts the cyclone's wind speed and classification, and returns instant intensity estimations and observation tracking history.

## 4. Key Features

- Multi-source satellite data ingestion
- Automated wind speed prediction
- Mathematical error reduction
- Instant disaster forecasting
- Live inference deployment

## 5. Technology Stack

- Frontend: Streamlit, HTML, CSS, SVG
- Backend: Python
- Machine Learning: PyTorch (torch.amp), NumPy, OpenCV, Swin-T, EfficientNet-B0
- Database: HDF5 (.h5) multidimensional arrays via h5py
- Deployment / Hardware: CloudGPU (FP16 optimized) 

## 6. Architecture

See [docs/architecture.md](docs/architecture.md).

```text
             User
              |
              v
Satellite Data Archive (HDF5)
              |
              v
  +-----------------------+
  |                       |
  v                       v
Ch.0 (BT/IR Image)     Ch.2 (Raw Visible Image)
  |                       |
  v                       v
Branch A (Swin-T)      Branch B (EfficientNet-B0)
  |                       |
  +-----------+-----------+
              |
              v
       Late Fusion (MLP)
              |
              v
 Intensity (Estimated Wind Speed)
```

## 7. Repository Structure

```text
SIH-26/
├── README.md
├── submission/
│   ├── PRESENTATION.md
│   └── DEMO.md
├── src/
│   └── app.py
│   └── backend.py
├── docs/
│   └── architecture.md
├── assets/
│   └── screenshots/
├── requirements.txt
└── LICENSE
```

### What goes where?

| Item | Location |
|---|---|
| Source code | `src/` |
| Architecture / technical documentation | `docs/` |
| Project screenshots / hardware photos | `assets/screenshots/` |
| Final PPT / presentation | `submission/` |
| Demo video link | `submission/DEMO.md` |
| Project overview | `README.md` |

## Deployed Web App

[Access N.E.T.R.A. Web App](https://sih-26-tech-wizards.streamlit.app/)

## 8. Final Presentation

[Presentation Deck (Google Drive)](https://drive.google.com/file/d/1C5QmsU-WvYQSqNAqFDryDQAV2HPiO4k1/view?usp=sharing)

## 9. Demo Video

[Watch Demo Video (Google Drive)](https://drive.google.com/file/d/1XMe3xrNfGSzVUhqXKs_Qpe06QqwuXWrR/view?usp=sharing)

## 10. Screenshots / Prototype Photos

[Project Dashboard](assets/screenshots/01-Dashboard.jpeg)

## 11. Installation

```bash
git clone <https://github.com/akshatbansal1087/SIH-26/>
cd <SIH-26>
pip install -r requirements.txt
```

## 12. Run

```bash
streamlit run app.py
```

## 13. Future Scope

- Trajectory Tracking: Add geographic forecasting to predict the cyclone's future path.
- Time-Series Modeling: Analyze continuous satellite image sequences to forecast rapid storm evolution once we get access to spontaneous data.
- Additional Sensors: Integrate Sea Surface Temperature and radar data alongside the current BT and RAW imagery.
