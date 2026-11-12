import { useCallback, useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import {
  Camera,
  Play,
  Square,
  RotateCcw,
  History,
  Info,
  Wifi,
} from "lucide-react";
import api from "../services/api";
import { useAuth } from "../context/AuthContext";
import PredictionBox from "../components/PredictionBox";
import "./LivePredict.css";

const INTERVAL_MS = 350;
const HISTORY_SIZE = 10;
const MIN_STABLE_COUNT = 6;
const CONFIDENCE_THRESHOLD = 75;
// react-webcam mirrored={false} sends the raw image. FastAPI uses flip=True,
// matching the Kaggle feature converter. Change only after testing orientation.
const CAMERA_MIRRORED = false;
const videoConstraints = { width: 640, height: 480, facingMode: "user" };

export default function LivePredict() {
  const webcamRef = useRef(null),
    timerRef = useRef(null),
    runningRef = useRef(false),
    sessionRef = useRef(0),
    historyRef = useRef([]),
    abortRef = useRef(null),
    sessionIdRef = useRef(null);
  if (!sessionIdRef.current) sessionIdRef.current = crypto.randomUUID();
  const { user } = useAuth();
  const [running, setRunning] = useState(false),
    [cameraReady, setCameraReady] = useState(false),
    [loading, setLoading] = useState(false),
    [status, setStatus] = useState("idle"),
    [message, setMessage] = useState("Waiting for camera"),
    [error, setError] = useState("");
  const [prediction, setPrediction] = useState(null),
    [confidence, setConfidence] = useState(0),
    [stableCount, setStableCount] = useState(0),
    [recent, setRecent] = useState([]);
  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);
  const resetResult = useCallback((nextStatus = "idle") => {
    historyRef.current = [];
    setPrediction(null);
    setConfidence(0);
    setStableCount(0);
    setStatus(nextStatus);
  }, []);
  const stop = useCallback(() => {
    runningRef.current = false;
    sessionRef.current += 1;
    clearTimer();
    abortRef.current?.abort();
    abortRef.current = null;
    setRunning(false);
    setLoading(false);
    resetResult("idle");
    setMessage("Recognition stopped");
  }, [clearTimer, resetResult]);
  const predictFrame = useCallback(
    async (session) => {
      if (!runningRef.current || session !== sessionRef.current) return;
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const image = webcamRef.current?.getScreenshot();
        if (!image) {
          resetResult("no_hand");
          setMessage("Waiting for camera frame");
          return;
        }
        setLoading(true);
        const response = await api.post(
          "/predict",
          { image },
          {
            signal: controller.signal,
            headers: { "X-Session-Id": sessionIdRef.current },
          },
        );
        if (!runningRef.current || session !== sessionRef.current) return;
        const data = response.data;
        if (!data?.success)
          throw new Error(data?.message || "Invalid prediction response");
        const conf = Number(data.confidence) || 0;
        if (
          data.status !== "detected" ||
          !data.prediction ||
          conf < CONFIDENCE_THRESHOLD
        ) {
          resetResult(data.status === "no_hand" ? "no_hand" : "low_confidence");
          setConfidence(data.status === "low_confidence" ? conf : 0);
          setMessage(
            data.status === "no_hand"
              ? "No hand detected — show your hand to the camera"
              : "Hold a clearer hand pose",
          );
          return;
        }
        const sign = String(data.prediction);
        historyRef.current = [...historyRef.current, sign].slice(-HISTORY_SIZE);
        const counts = new Map();
        for (const item of historyRef.current)
          counts.set(item, (counts.get(item) || 0) + 1);
        const [winner, votes] = [...counts.entries()].sort(
          (a, b) => b[1] - a[1],
        )[0];
        setConfidence(conf);
        setStableCount(votes);
        if (votes >= MIN_STABLE_COUNT) {
          setPrediction(winner);
          setStatus("detected");
          setMessage(`Stable sign: ${winner}`);
          setRecent((prev) => {
            if (prev[0]?.sign === winner) return prev;
            return [
              { sign: winner, time: new Date().toLocaleTimeString() },
              ...prev,
            ].slice(0, 8);
          });
        } else {
          setPrediction(null);
          setStatus("stabilizing");
          setMessage(`Stabilizing: ${votes}/${MIN_STABLE_COUNT} votes`);
        }
      } catch (err) {
        if (
          !runningRef.current ||
          session !== sessionRef.current ||
          controller.signal.aborted
        )
          return;
        resetResult("error");
        setMessage("Prediction service unavailable");
        setError(
          err.response?.data?.message ||
            err.response?.data?.detail ||
            err.message ||
            "Could not connect to the backend",
        );
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
        if (runningRef.current && session === sessionRef.current) {
          setLoading(false);
          timerRef.current = setTimeout(
            () => predictFrame(session),
            INTERVAL_MS,
          );
        }
      }
    },
    [resetResult],
  );
  const start = () => {
    if (!cameraReady || runningRef.current) return;
    clearTimer();
    abortRef.current?.abort();
    runningRef.current = true;
    sessionRef.current += 1;
    historyRef.current = [];
    setRunning(true);
    setError("");
    resetResult("stabilizing");
    setMessage("Starting recognition...");
    predictFrame(sessionRef.current);
  };
  useEffect(
    () => () => {
      runningRef.current = false;
      sessionRef.current += 1;
      clearTimer();
      abortRef.current?.abort();
    },
    [clearTimer],
  );
  const handleCameraError = (err) => {
    setCameraReady(false);
    stop();
    setMessage("Camera unavailable");
    setError(err?.message || "Allow camera access and refresh the page");
  };
  return (
    <div className="live-page">
      <div className="page-heading">
        <span className="eyebrow">AI RECOGNITION STUDIO</span>
        <h1>Live Sign Recognition</h1>
        <p>
          Position your hand inside the camera view and hold a gesture steady.
          SignAI analyzes frames using your trained model.
        </p>
      </div>
      <div className="live-toolbar">
        <span className={`live-indicator ${running ? "active" : ""}`}>
          <span />
          {running
            ? "Recognition active"
            : cameraReady
              ? "Camera ready"
              : "Camera initializing"}
        </span>
        <span className="live-model-tag">
          <Wifi size={15} /> TensorFlow · 126 features
        </span>
      </div>
      <div className="live-grid">
        <div className="camera-column">
          <section className="camera-card panel">
            <div className="camera-card-header">
              <span>
                <Camera size={17} /> Camera feed
              </span>
              <span className="camera-resolution">640 × 480</span>
            </div>
            <div className="camera-stage">
              <Webcam
                ref={webcamRef}
                audio={false}
                mirrored={CAMERA_MIRRORED}
                screenshotFormat="image/jpeg"
                screenshotQuality={0.85}
                videoConstraints={videoConstraints}
                className="camera-video"
                onUserMedia={() => {
                  setCameraReady(true);
                  setError("");
                  setMessage("Camera ready — start recognition");
                }}
                onUserMediaError={handleCameraError}
              />
              <div className="camera-corner top-left" />
              <div className="camera-corner top-right" />
              <div className="camera-corner bottom-left" />
              <div className="camera-corner bottom-right" />
              <span className="camera-overlay-label">
                {running ? "● ANALYZING LIVE" : "CAMERA PREVIEW"}
              </span>
            </div>
            <div className="camera-actions">
              <button
                className="btn btn-primary"
                onClick={start}
                disabled={!cameraReady || running}
              >
                <Play size={17} /> Start Recognition
              </button>
              <button
                className="btn btn-danger"
                onClick={stop}
                disabled={!running}
              >
                <Square size={17} /> Stop
              </button>
              <button
                className="btn btn-outline"
                onClick={() => {
                  historyRef.current = [];
                  setRecent([]);
                  resetResult(running ? "stabilizing" : "idle");
                  setMessage("Session cleared");
                }}
              >
                <RotateCcw size={17} /> Clear
              </button>
            </div>
            <div className="camera-feedback" role="status">
              <span className={loading ? "loading-pulse" : ""} />
              {loading ? "Processing frame..." : message}
            </div>
            {error && (
              <div className="alert" role="alert">
                {error}
              </div>
            )}
          </section>
          <section className="tips-panel panel">
            <Info size={19} />
            <div>
              <strong>For reliable recognition</strong>
              <p>
                Use even lighting, keep your hand fully visible, and hold one
                pose for several predictions. This model recognizes static
                gestures, not continuous sign-language sentences.
              </p>
            </div>
          </section>
        </div>
        <div className="live-side">
          <PredictionBox
            prediction={prediction}
            confidence={confidence}
            status={status}
            stableCount={stableCount}
            requiredCount={MIN_STABLE_COUNT}
          />
          <section className="recent-panel panel">
            <div className="recent-heading">
              <h3>
                <History size={18} /> Session predictions
              </h3>
              <span>{recent.length} recent</span>
            </div>
            {recent.length ? (
              <div className="recent-list">
                {recent.map((item, i) => (
                  <div key={`${item.time}-${i}`}>
                    <strong>{item.sign}</strong>
                    <span>{item.time}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="recent-empty">
                Stable recognized signs will appear here during this session.
              </p>
            )}
            {!user && (
              <p className="recent-footnote">
                Log in to save eligible predictions to your account history.
              </p>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
