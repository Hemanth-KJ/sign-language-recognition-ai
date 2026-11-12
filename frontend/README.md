# SignAI Frontend — Professional UI

## Requirements
- Node.js and npm
- Node/Express backend on `http://localhost:5000`
- FastAPI model service on `http://127.0.0.1:8000`
- MongoDB connected to Express

## Install
```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

If replacing an existing frontend, copy the `src` files into your existing Vite project, and keep its existing package.json if dependencies are installed. Install missing packages with `npm install axios react-router-dom react-webcam lucide-react`.

## Recognition orientation
`LivePredict.jsx` uses `mirrored={false}` and FastAPI `extract_hand_features(..., flip=True)`, matching the Kaggle conversion's flip. This avoids assuming react-webcam screenshot mirroring. Compare webcam results against `predict.py` on a known sign and verify left/right-hand landmark slots. If you change orientation, change only one flip in the pipeline.

## Prediction logic
- Requests are sequential; no overlapping inference requests.
- React collects 10 accepted predictions and displays a sign after 6 matching votes, like `predict.py`.
- `no_hand` and `low_confidence` clear stabilization history.
- A new AbortController and session ID prevent old responses from updating the UI after stop/restart.
- An `X-Session-Id` header enables backend cooldown saving for anonymous sessions; authenticated requests use the user ID.

## Important limitations
- A softmax score is not a calibrated measure of correctness.
- Real webcam accuracy is not guaranteed by a high held-out Kaggle score.
- J and Z are motion-based ASL letters; static hand features alone cannot reliably model their movements.
- `Sign Library` stores metadata only and does not retrain the model.
- Session predictions are local to the page; backend history is saved only for eligible detected predictions and is not displayed by this frontend.
