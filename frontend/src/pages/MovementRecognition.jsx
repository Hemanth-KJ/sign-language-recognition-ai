
import { useEffect, useRef, useState } from "react";
import "./MovementRecognition.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://localhost:5000";

const CAPTURE_INTERVAL = 200;

export default function MovementRecognition() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const sessionRef = useRef(null);
  const busyRef = useRef(false);
  const activeRef = useRef(false);
  const generationRef = useRef(0);

  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("idle");
  const [sign, setSign] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [frames, setFrames] = useState(0);
  const [error, setError] = useState("");

  async function sendFrame() {
    if (!activeRef.current || busyRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const sessionId = sessionRef.current;

    if (!video || !canvas || !sessionId) return;
    if (video.readyState < 2) return;

    const generation = generationRef.current;
    busyRef.current = true;

    try {
      canvas.width = 320;
      canvas.height = 240;

      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Canvas unavailable");

      ctx.drawImage(video, 0, 0, 320, 240);

      const image = canvas.toDataURL("image/jpeg", 0.7);

      const response = await fetch(
        `${API_URL}/api/movement/predict-image`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ sessionId, image }),
        }
      );

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(
          result.message || result.detail || "Recognition failed"
        );
      }

      if (
        !activeRef.current ||
        generation !== generationRef.current
      ) {
        return;
      }

      const data = result.data;

      setStatus(data.status);
      setSign(data.sign);
      setConfidence(data.confidence);
      setFrames(data.framesCollected);
      setError("");
    } catch (err) {
      if (
        activeRef.current &&
        generation === generationRef.current
      ) {
        setError(err.message);
        setStatus("error");
      }
    } finally {
      busyRef.current = false;
    }
  }

  async function startCamera() {
    if (activeRef.current) return;

    const generation = ++generationRef.current;

    try {
      setError("");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          facingMode: "user",
        },
        audio: false,
      });

      if (generation !== generationRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;
      sessionRef.current = crypto.randomUUID();

      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      if (generation !== generationRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      activeRef.current = true;
      setRunning(true);
      setStatus("collecting");
      setSign(null);
      setConfidence(0);
      setFrames(0);

      intervalRef.current = setInterval(
        sendFrame,
        CAPTURE_INTERVAL
      );
    } catch (err) {
      setError(err.message);
      setStatus("error");
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }

  async function stopCamera() {
    activeRef.current = false;
    generationRef.current += 1;

    clearInterval(intervalRef.current);
    intervalRef.current = null;

    streamRef.current?.getTracks().forEach((track) => {
      track.stop();
    });
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    const sessionId = sessionRef.current;
    sessionRef.current = null;

    setRunning(false);
    setStatus("idle");
    setSign(null);
    setConfidence(0);
    setFrames(0);

    if (sessionId) {
      try {
        await fetch(`${API_URL}/api/movement/reset`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ sessionId }),
        });
      } catch (err) {
        console.error("Session reset error:", err);
      }
    }
  }

  useEffect(() => {
    return () => {
      activeRef.current = false;
      generationRef.current += 1;
      clearInterval(intervalRef.current);
      streamRef.current?.getTracks().forEach((track) => {
        track.stop();
      });
    };
  }, []);

  const statusText = {
    idle: "Camera stopped",
    waiting: "Show your hands to the camera",
    collecting: "Collecting movement frames",
    analyzing: "Analyzing your gesture",
    recognized: "Gesture recognized",
    unknown: "Unknown gesture",
    error: "Recognition error",
  };

  return (
    <div className="movement-page">
      <div className="movement-container">
        <div className="movement-heading">
          <span className="movement-eyebrow">
            SignAI / Dynamic Recognition
          </span>

          <h1>Live Movement Recognition</h1>

          <p>
            Recognize dynamic sign language gestures
            using MediaPipe and an LSTM neural network.
          </p>
        </div>

        <div className="movement-layout">
          <div className="movement-camera-card">
            <div className="movement-card-header">
              <h2>Live Camera</h2>

              <span
                className={
                  running
                    ? "movement-indicator active"
                    : "movement-indicator"
                }
              >
                {running ? "Live" : "Offline"}
              </span>
            </div>

            <div className="movement-video-wrapper">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="movement-video"
              />

              {!running && (
                <div className="movement-placeholder">
                  <span>Camera is off</span>
                  <p>Start the camera to recognize gestures</p>
                </div>
              )}
            </div>

            <canvas
              ref={canvasRef}
              style={{ display: "none" }}
            />

            <div className="movement-camera-controls">
              {!running ? (
                <button
                  className="movement-btn primary"
                  onClick={startCamera}
                >
                  Start Recognition
                </button>
              ) : (
                <button
                  className="movement-btn secondary"
                  onClick={stopCamera}
                >
                  Stop Recognition
                </button>
              )}
            </div>
          </div>

          <div className="movement-result-card">
            <h2>Recognition Result</h2>

            <div className="movement-sign-display">
              <span>Detected Gesture</span>
              <strong>{sign || "—"}</strong>
            </div>

            <div className="movement-metric">
              <span>Confidence</span>
              <strong>
                {(confidence * 100).toFixed(1)}%
              </strong>
            </div>

            <div className="movement-confidence-track">
              <div
                className="movement-confidence-fill"
                style={{
                  width: `${Math.min(
                    100,
                    Math.max(0, confidence * 100)
                  )}%`,
                }}
              />
            </div>

            <div className="movement-metric">
              <span>Frames Collected</span>
              <strong>{frames}/30</strong>
            </div>

            <div className="movement-metric">
              <span>Status</span>
              <strong>{statusText[status] || status}</strong>
            </div>

            <div className="movement-supported">
              <h3>Supported Gestures</h3>
              <div>
                <span>HELLO</span>
                <span>THANK_YOU</span>
              </div>
            </div>
          </div>
        </div>

        {error && (
          <div className="movement-error">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
