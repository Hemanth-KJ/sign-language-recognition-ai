# SignAI — Sign Language Recognition Using Neural Networks

> A full-stack sign language recognition project that uses computer vision, hand landmarks, and neural networks to recognize static hand signs and movement-based gestures.

SignAI brings together a **React frontend**, a **Node.js/Express backend**, and a **Python AI service**. A camera captures hand movements; MediaPipe extracts hand landmarks; trained TensorFlow models interpret those features; and the application displays the predicted sign.

This README is written for **first-time visitors**. You do not need to know the repository in advance: start with [What is SignAI?](#what-is-signai), then read [How it works](#how-signai-works), and follow [Getting started](#getting-started) to explore it locally.

> **Project status:** Static alphabet recognition and dynamic gesture recognition have been implemented. An expanded WLASL vocabulary model has been trained and evaluated as a **candidate**, but it should not be assumed to be integrated into the live application.

## Table of contents

1. [What is SignAI?](#what-is-signai)
2. [The problem and our approach](#the-problem-and-our-approach)
3. [What you can do with SignAI](#what-you-can-do-with-signai)
4. [How SignAI works](#how-signai-works)
5. [Technology stack](#technology-stack)
6. [Repository structure](#repository-structure)
7. [Understanding the three services](#understanding-the-three-services)
8. [Static vs. dynamic recognition](#static-vs-dynamic-recognition)
9. [Datasets and model artifacts](#datasets-and-model-artifacts)
10. [Getting started](#getting-started)
11. [Running the application](#running-the-application)
12. [Training and testing](#training-and-testing)
13. [API and data flow](#api-and-data-flow)
14. [Evaluation results](#evaluation-results)
15. [Limitations and future work](#limitations-and-future-work)
16. [Troubleshooting](#troubleshooting)
17. [Security and repository hygiene](#security-and-repository-hygiene)
18. [Contributing](#contributing)

## What is SignAI?

**SignAI** is a sign language recognition application designed to help explore more accessible communication between people who use hand signs and people who may not understand them. The application analyzes hand positions and movements from camera input and produces a predicted sign label.

It is also an educational deep-learning project: the repository contains code for **collecting data**, **extracting features**, **preparing datasets**, **training models**, **running predictions**, and **connecting the models to a web application**.

### Who is this project for?

- **First-time users:** Try recognition through the web interface.
- **Students:** Learn how computer vision and neural networks fit into a full-stack application.
- **Developers:** Explore the separation between frontend, backend, and model inference.
- **Researchers:** Examine the difference between static image classification and temporal gesture recognition.

## The problem and our approach

Many sign-language gestures are visually subtle. A recognition system must account for hand position, finger shape, orientation, and — for moving signs — changes over time. Camera backgrounds, lighting, occlusion, and differences between signers make the task harder.

SignAI approaches this in two parts:

1. **Static recognition:** Extract hand landmarks from a frame and classify the hand configuration.
2. **Dynamic recognition:** Collect a sequence of landmark features and classify the movement pattern using an LSTM neural network.

The application uses **landmark features** rather than sending raw images directly to the static classifier. This reduces the model input to a consistent numerical representation, but also means recognition depends on successful hand detection and preprocessing.

## What you can do with SignAI

The repository includes code and UI components for:

- **User authentication:** Login and registration pages, authentication context, and backend authentication modules.
- **Static sign prediction:** Webcam-based alphabet/sign classification through a trained model.
- **Dynamic gesture recognition:** Recognition of movement-based signs from a sequence of hand landmarks.
- **Sign collection:** UI and Python scripts for capturing additional sign samples.
- **Dataset preparation:** Conversion and selection utilities for Kaggle alphabet data and WLASL videos.
- **Model experimentation:** Scripts for training and testing neural-network models.
- **Prediction display:** A frontend prediction component and dedicated live-prediction pages.

> The presence of a module in the repository does not by itself prove that every route or screen is production-ready. Run the services and verify the relevant workflow in your local environment.

## How SignAI works

```text
                          USER / WEBCAM
                               |
                               v
                     React frontend (UI)
                               |
                Authentication / app requests
                               v
                     Node.js / Express API
                               |
                    Prediction integration
                               v
                       Python AI service
                               |
                      MediaPipe Hands
                               |
                   21 landmarks per hand
                               |
                2 hands × 21 × 3 = 126 values
                               |
                    +----------+----------+
                    |                     |
                    v                     v
             Static classifier     Dynamic LSTM model
              Single feature       30-frame sequence
                vector               (30 × 126)
                    |                     |
                    +----------+----------+
                               |
                               v
                    Predicted sign / status
                               |
                               v
                    React recognition UI
```

**In simple terms:** the camera sees a hand; MediaPipe turns it into numbers; the trained model uses those numbers to predict a sign; the interface displays the result.

### Why 126 features?

MediaPipe provides **21 landmarks** for each detected hand. Each landmark has **x, y, and z** coordinates. SignAI's two-hand feature representation therefore contains:

```text
2 hands × 21 landmarks × 3 coordinates = 126 features
```

The feature-extraction code normalizes landmarks relative to the wrist and hand scale to reduce sensitivity to absolute image position and hand size.

## Technology stack

| Layer | Technologies | Purpose |
|---|---|---|
| Frontend | React, JSX, CSS | Pages, camera-driven recognition interface, prediction display |
| Backend | Node.js, Express | Application APIs, authentication, prediction and movement orchestration |
| AI service | Python, FastAPI | Model inference and movement prediction endpoints |
| Computer vision | OpenCV, MediaPipe | Camera processing and hand landmark extraction |
| Deep learning | TensorFlow / Keras | Static classification and LSTM-based movement recognition |
| Data processing | NumPy | Landmark arrays and training/test sequences |
| Persistence | Backend `config/db.js` and model modules | Application data integration; inspect configuration for the actual database provider |
| Development | VS Code, Git, GitHub | Local development and version control |

## Repository structure

The tree below follows the **actual folder and file names visible in the shared project screenshots**. Additional files or directories may exist beyond the screenshot boundaries.

```text
sign-language-recognition/
|
|-- ai-service/                         # Python computer vision + AI
|   |-- app.py                          # FastAPI application entry point
|   |-- capture.py                      # Capture static sign samples
|   |-- capture_movement.py             # Capture dynamic gesture sequences
|   |-- check_wlasl_download.py         # Inspect WLASL download status
|   |-- check_wlasl_numpy.py            # Inspect converted NumPy samples
|   |-- convert_wlasl.py                # Convert WLASL videos to features
|   |-- hand_features.py                # Static hand feature extraction
|   |-- movement_features.py            # Dynamic hand feature extraction
|   |-- movement_routes.py              # FastAPI movement endpoints
|   |-- movement_service.py             # Dynamic prediction/session logic
|   |-- predict.py                      # Static sign prediction
|   |-- predict_movement.py             # Dynamic prediction script
|   |-- prepare_kaggle_dataset.py       # Kaggle dataset preparation
|   |-- requirements.txt                # Python dependencies
|   |-- test_candidate.py               # Candidate model evaluation
|   |-- train.py                        # Static model training
|   |-- train_lstm.py                   # LSTM model training
|   |-- wlasl_download.py               # WLASL download utility
|   |-- wlasl_select.py                 # WLASL sample/class selection
|   `-- venv/                           # Local virtual environment (do not commit)
|
|-- backend/                            # Node.js / Express API
|   |-- config/
|   |   `-- db.js                        # Database connection/configuration
|   |-- controllers/
|   |   |-- authController.js
|   |   |-- movementController.js
|   |   |-- predictController.js
|   |   `-- signController.js
|   |-- middleware/
|   |   `-- authMiddleware.js
|   |-- models/
|   |   |-- Prediction.js
|   |   |-- SignModelTemp.js
|   |   `-- User.js
|   |-- routes/
|   |   |-- auth.js
|   |   |-- movementRoutes.js
|   |   |-- predict.js
|   |   `-- sign.js
|   |-- services/
|   |   `-- movementService.js
|   |-- app.js                            # Express application entry point
|   |-- package.json
|   |-- package-lock.json
|   `-- .env                              # Local secrets (do not commit)
|
|-- frontend/                           # React web interface
|   |-- src/
|   |   |-- components/
|   |   |   |-- Navbar.jsx
|   |   |   |-- PredictionBox.jsx
|   |   |   `-- PredictionBox.css
|   |   |-- context/
|   |   |   `-- AuthContext.jsx
|   |   |-- layout/
|   |   |   |-- Layout.jsx
|   |   |   `-- Layout.css
|   |   |-- pages/
|   |   |   |-- AddSign.jsx / AddSign.css
|   |   |   |-- Auth.css
|   |   |   |-- Home.jsx / Home.css
|   |   |   |-- LivePredict.jsx / LivePredict.css
|   |   |   |-- Login.jsx
|   |   |   |-- Register.jsx
|   |   |   `-- MovementRecognition.jsx / MovementRecognition.css
|   |   |-- services/
|   |   |   `-- api.js
|   |   |-- App.jsx
|   |   `-- App.css / index.css
|   `-- package.json                     # Frontend scripts and dependencies
|
|-- datasets/                            # Training / evaluation data
|   |-- hand_features/                   # Static landmark features
|   `-- kaggle/
|       `-- asl_alphabet_test/
|           |-- A_test.jpg
|           |-- B_test.jpg
|           `-- ...
|
|-- models/                              # Saved neural-network models
|   |-- hand_model/
|   |   |-- sign_model.keras
|   |   `-- labels.npy
|   |-- movement_model/                  # Active movement model artifacts
|   `-- movement_model_candidate/        # Expanded experimental model
|
`-- README.md
```

**A quick navigation guide:**

| If you want to... | Start here |
|---|---|
| Understand the user experience | `frontend/src/pages/Home.jsx` |
| See the alphabet prediction UI | `frontend/src/pages/LivePredict.jsx` |
| Explore movement recognition | `frontend/src/pages/MovementRecognition.jsx` |
| Understand frontend API calls | `frontend/src/services/api.js` |
| Understand the Node server | `backend/app.js` |
| Find the backend endpoints | `backend/routes/` |
| Understand Python inference | `ai-service/app.py` |
| Understand hand feature extraction | `ai-service/hand_features.py` and `movement_features.py` |
| Find the dynamic prediction logic | `ai-service/movement_service.py` |
| Explore model training | `ai-service/train.py` and `train_lstm.py` |
| Explore dataset preparation | `prepare_kaggle_dataset.py` and `convert_wlasl.py` |

## Understanding the three services

### 1. Frontend — what the user sees

The **frontend** is the browser application. Its `pages/` directory separates different user workflows:

- **`Home.jsx`** — entry page for visitors.
- **`Login.jsx` and `Register.jsx`** — authentication screens.
- **`LivePredict.jsx`** — live sign prediction interface.
- **`MovementRecognition.jsx`** — movement-based recognition interface.
- **`AddSign.jsx`** — sign data entry/collection workflow.

Shared UI pieces live in `components/`, login state is handled in `context/AuthContext.jsx`, and API requests are centralized in `services/api.js`.

### 2. Backend — how the application is coordinated

The **backend** uses a conventional Express structure:

- **Routes** receive HTTP requests.
- **Controllers** handle request-level logic.
- **Services** contain reusable application logic.
- **Models** represent application data.
- **Middleware** provides cross-cutting checks such as authentication.
- **`config/db.js`** configures the database connection.

For example, a request related to movement recognition may pass through `routes/movementRoutes.js`, `controllers/movementController.js`, and `services/movementService.js` before a result is returned to the frontend. Check those files for the exact live endpoint and forwarding behavior.

### 3. AI service — how a sign becomes a prediction

The **AI service** runs the computer-vision and deep-learning code:

- **`hand_features.py`** converts detected static hand landmarks into a 126-value vector.
- **`movement_features.py`** extracts features for each frame of a dynamic gesture.
- **`movement_service.py`** manages temporal prediction, confidence filtering, and prediction state.
- **`movement_routes.py`** exposes movement-related FastAPI endpoints.
- **`predict.py`** and **`predict_movement.py`** provide prediction workflows.
- **`train.py`** and **`train_lstm.py`** contain training logic.

## Static vs. dynamic recognition

| Aspect | Static recognition | Dynamic recognition |
|---|---|---|
| What is recognized? | A hand shape or pose | A sign involving movement over time |
| Model input | One 126-feature landmark vector | A sequence of 30 frames × 126 features |
| Core model | Saved TensorFlow classifier | LSTM neural network |
| Key files | `hand_features.py`, `predict.py`, `train.py` | `movement_features.py`, `movement_service.py`, `train_lstm.py` |
| Example | Alphabet hand shape | `HELLO`, `THANK_YOU` |

### Dynamic prediction stability

The dynamic recognition implementation includes mechanisms intended to reduce unstable output, such as confidence thresholds, repeated prediction checks, session tracking, and movement-related gating. These mechanisms help prevent a label from changing on every frame, although their effectiveness must be evaluated in real use.

## Datasets and model artifacts

### Kaggle ASL Alphabet

The repository includes a Kaggle alphabet test-image directory with files such as `A_test.jpg`, `B_test.jpg`, and so on. These are **images**, not already-extracted landmark vectors. To evaluate the static model on them, the test pipeline must first run MediaPipe using the **same preprocessing assumptions as training**.

### Hand landmark feature dataset

The `datasets/hand_features/` directory is used for static sign feature data. The exact file layout and saved array shapes should be checked against `train.py` before running a test script. Do not assume the files are organized as `A/1.npy`, `B/1.npy`, etc. without verifying.

### WLASL dynamic gestures

**WLASL (Word-Level American Sign Language)** is used for experiments with word-level movement recognition. Supporting scripts select/download clips and convert videos into fixed-length NumPy feature sequences. The dynamic model works with **30-frame sequences** of hand features.

### Model storage

| Artifact | Purpose |
|---|---|
| `models/hand_model/sign_model.keras` | Trained static classifier |
| `models/hand_model/labels.npy` | Static model class-label mapping |
| `models/movement_model/` | Active dynamic recognition model and labels |
| `models/movement_model_candidate/` | Experimental expanded-vocabulary model |

Model files and dataset archives may be large. Their availability in a fresh clone depends on what has actually been committed to Git or provided through a separate download.

## Getting started

### Prerequisites

Install:

- **Python 3.10** (the development environment shown uses Python 3.10)
- **Node.js and npm** (use a version compatible with the `frontend` and `backend` packages)
- **Git**
- **A webcam** for live recognition
- **VS Code** or another editor (optional)
- Access to the **database configured in `backend/config/db.js`**, if required for the backend workflows

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd sign-language-recognition
```

Replace `<YOUR_REPOSITORY_URL>` with the real GitHub clone URL.

### 2. Set up the Python AI service (Windows PowerShell)

```powershell
cd ai-service
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the virtual environment already exists locally, activate it rather than creating it again. Do not commit `venv/` to Git.

### 3. Set up the Node.js backend

Open a **new terminal** from the repository root:

```powershell
cd backend
npm install
```

Create/configure `backend/.env` using the variables actually read by `backend/app.js`, `backend/config/db.js`, and the controller/service modules. **Do not copy credentials from someone else's machine or commit the `.env` file.**

> The exact database provider, environment-variable names, and backend port must be verified in the source files. They are intentionally not guessed here.

### 4. Set up the React frontend

Open another terminal from the repository root:

```powershell
cd frontend
npm install
```

Inspect `frontend/package.json` for available scripts and `frontend/src/services/api.js` for the API base URL. Point the frontend to your locally running backend as required.

## Running the application

The system normally uses **three terminals**, one per service.

### Terminal 1 — AI service

```powershell
cd ai-service
.\venv\Scripts\Activate.ps1
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

FastAPI interactive API documentation is typically available at:

```text
http://127.0.0.1:8000/docs
```

### Terminal 2 — backend

```powershell
cd backend
npm start
```

This command requires a `start` script in `backend/package.json`. If no `start` script exists, inspect the scripts or use the appropriate command for `app.js` (for example, `node app.js`).

### Terminal 3 — frontend

```powershell
cd frontend
npm run dev
```

This command requires a `dev` script in `frontend/package.json`. Open the local URL printed by the frontend development server.

### First-time walkthrough

1. Start the AI service, backend, and frontend.
2. Open the frontend URL in your browser.
3. Explore the Home page and, if required, register or log in.
4. Open **Live Prediction** for static sign recognition.
5. Allow camera access and position your hand clearly in frame.
6. Open **Movement Recognition** to try the dynamic gesture workflow.
7. Observe the predicted label and confidence/status feedback.

> Webcam access can require a secure context depending on the browser and deployment environment. `localhost` is generally suitable for local development.

## Training and testing

### Static sign training

The static model's training logic is in:

```text
ai-service/train.py
```

Before retraining, inspect the script for dataset paths, expected array shapes, and how labels are encoded. The model used for prediction expects a **126-feature input**.

### Dynamic sign training

The LSTM training logic is in:

```text
ai-service/train_lstm.py
```

Dynamic samples are prepared as 30-frame sequences. Relevant scripts include:

```text
ai-service/capture_movement.py
ai-service/wlasl_select.py
ai-service/wlasl_download.py
ai-service/convert_wlasl.py
ai-service/check_wlasl_numpy.py
```

### Candidate model evaluation

```text
ai-service/test_candidate.py
```

Use this script to inspect/evaluate the expanded movement model. Check its command-line arguments and configured paths before running it.

### External Kaggle alphabet testing

An external test on the Kaggle `asl_alphabet_test` images produced the following measured results:

| Metric | Result |
|---|---:|
| Total image files | 28 |
| Supported images | 27 |
| Hand detections | 18 |
| Correct predictions | 1 |
| Incorrect predictions on detected images | 17 |
| No hand detected | 9 |
| Hand detection rate | 66.67% |
| Accuracy on detected images | 5.56% |
| **End-to-end accuracy on supported images** | **3.70%** |

This is an **external dataset result**, not the same as the model's training or original validation accuracy. It highlights a generalization/preprocessing problem that should be investigated. A previously reported **99% alphabet accuracy** has not been identified here as training, validation, or held-out test accuracy, so it should not be represented as an independently verified result without the corresponding evaluation logs.

### WLASL candidate evaluation

An expanded dynamic candidate was evaluated on **15 held-out WLASL sequences**:

| Metric | Result |
|---|---:|
| Training sequences | 40 |
| Validation sequences | 15 |
| Held-out test sequences | 15 |
| Correct held-out predictions | 13 |
| **Held-out test accuracy** | **86.67%** |

This result applies to the **candidate model**, not necessarily the model currently served by the live movement-recognition interface. The test set is small; further signer-independent and real-world testing is needed.

## API and data flow

### Backend route files

The Express backend organizes its routes under:

```text
backend/routes/auth.js
backend/routes/predict.js
backend/routes/sign.js
backend/routes/movementRoutes.js
```

For the exact HTTP methods, URL paths, request bodies, and authentication requirements, open these route files and inspect the handlers. The filenames identify the areas of responsibility, but not the exact public API contract.

### Python movement API

The Python movement router uses the prefix:

```text
/api/movement
```

Implemented route names in the shared code include:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/movement/health` | Check movement service status |
| POST | `/api/movement/predict` | Submit one frame's 126 features and session information |
| POST | `/api/movement/predict-image` | Submit an encoded webcam frame for landmark extraction and prediction |
| POST | `/api/movement/reset` | Reset a movement prediction session |

The `/predict` workflow uses a session identifier, hand-detection information, and a feature vector. Refer to `movement_routes.py` for the exact request schema and response fields. The interactive documentation at `/docs` is the easiest way to inspect the live FastAPI contract.

## Evaluation results

The project contains **two distinct recognition tasks** with different evaluation settings:

| Recognition task | Evaluation | Result | Interpretation |
|---|---|---|---|
| Static alphabet | Reported original model performance | 99% (evaluation split not verified) | Requires training/validation/test clarification |
| Static alphabet | External Kaggle image test | 3.70% end-to-end | Poor transfer to this small external image set |
| Dynamic WLASL candidate | Held-out 15-sequence test | 86.67% | Promising preliminary candidate result |

**Why the numbers differ:** The models, datasets, preprocessing paths, and evaluation populations are different. They must not be combined into a single “SignAI accuracy” number.

## Limitations and future work

### Current limitations

- Hand detection can fail on difficult lighting, occlusion, or unfamiliar image conditions.
- External Kaggle alphabet images currently produce poor end-to-end accuracy.
- Dynamic prediction is limited to the vocabulary supported by the **active** movement model.
- The expanded WLASL model remains a candidate pending integration and broader validation.
- Small test sets cannot establish reliable performance across many signers and environments.
- The system predicts supported labels; it is **not yet a general continuous sign-language translator**.

### Planned improvements

- Improve consistency between training-time and test-time feature extraction.
- Expand and diversify static and dynamic datasets.
- Evaluate on signer-independent, held-out test sets.
- Integrate the expanded WLASL model after validation.
- Add continuous multi-sign recognition and sentence formation.
- Explore text-to-speech output and mobile-friendly deployment.
- Add repeatable automated tests and documented model benchmarks.

## Troubleshooting

**`ModuleNotFoundError` while starting Uvicorn**  
Run the command from the `ai-service` directory. In the screenshot provided, the Python modules are **flat files directly under `ai-service/`**, so imports such as `from hand_features import ...` are appropriate. Do not change them to `from static.hand_features import ...` unless you actually move the files into a `static/` package.

**`No valid 126-feature samples found`**  
A test script may expect a different `.npy` directory layout from the one on disk. Inspect the actual feature files and the loading logic in `train.py`. Kaggle `.jpg` images must be processed through MediaPipe first.

**Model file not found**  
Verify that the saved `.keras` file and label mapping exist at the paths used in the code. Relative paths can behave differently depending on the current working directory.

**Webcam is blank or unavailable**  
Allow browser camera permission, close applications that are already using the camera, and try running on `localhost`.

**Frontend cannot reach the backend**  
Check the frontend API base URL, backend port, backend startup logs, and CORS settings.

**TensorFlow oneDNN / CPU optimization messages**  
These are usually informational startup messages, not failures. Read the **final traceback line** if the process exits.

**Low accuracy on Kaggle images**  
Check hand mirroring, landmark normalization, class-label mapping, MediaPipe detection success, and differences between training and test images before drawing conclusions or retraining.

## Security and repository hygiene

Do **not** commit:

```gitignore
# Python
ai-service/venv/
**/__pycache__/
*.pyc

# Node
**/node_modules/

# Secrets
**/.env
**/.env.*
!**/.env.example

# Generated files
results/
```

Keep any dataset, model, or generated-output exclusions aligned with your intended distribution strategy. If a model is needed for first-time setup but is not committed, provide a legitimate download link and explain where to place it.

## Contributing

Contributions are welcome, particularly for improving dataset quality, inference reliability, test coverage, accessibility, and documentation.

1. Fork the repository and create a feature branch.
2. Keep changes focused and document any new dependencies.
3. Add or update tests where possible.
4. Do not commit credentials, private user data, or unlicensed datasets.
5. Open a pull request explaining the change and how you tested it.

## Project summary

SignAI is both an **assistive-technology prototype** and a **full-stack machine-learning learning project**. It demonstrates how a camera, MediaPipe hand landmarks, TensorFlow models, FastAPI, Express, and React can work together to recognize static and dynamic signs.

The repository is a starting point for further work in sign-language recognition, model evaluation, accessible interfaces, and real-time computer vision.

---

**New to the codebase?** Start with `frontend/src/pages/Home.jsx`, then `backend/app.js`, then `ai-service/app.py`. After that, read `hand_features.py` and `movement_service.py` to understand the two recognition pipelines.
