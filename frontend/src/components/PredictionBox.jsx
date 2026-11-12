import { CheckCircle2, Activity, Hand, AlertCircle } from "lucide-react";
import "./PredictionBox.css";
export default function PredictionBox({
  prediction = null,
  confidence = 0,
  status = "idle",
  stableCount = 0,
  requiredCount = 6,
}) {
  const value = Number.isFinite(Number(confidence))
    ? Math.max(0, Math.min(100, Number(confidence)))
    : 0;
  const detected = status === "detected" && Boolean(prediction);
  const label =
    status === "no_hand"
      ? "No hand detected"
      : status === "low_confidence"
        ? "Low confidence"
        : status === "stabilizing"
          ? "Analyzing gesture"
          : status === "detected"
            ? "Stable prediction"
            : status === "error"
              ? "Service error"
              : "Waiting for recognition";
  return (
    <div className="result-panel panel">
      <div className="result-top">
        <div>
          <span className="eyebrow">MODEL OUTPUT</span>
          <h2>Recognition result</h2>
        </div>
        <span className={`result-status ${detected ? "is-detected" : ""}`}>
          {detected ? <CheckCircle2 size={15} /> : <Activity size={15} />}{" "}
          {label}
        </span>
      </div>
      <div className="result-main">
        <div className="result-letter">
          {detected ? (
            prediction === "space" ? (
              "SPACE"
            ) : prediction === "del" ? (
              "DELETE"
            ) : (
              prediction
            )
          ) : (
            <Hand size={64} strokeWidth={1.1} />
          )}
        </div>
        <div className="result-hint">
          {detected
            ? "Sign recognized"
            : "Hold a clear hand pose in front of the camera"}
        </div>
      </div>
      <div className="result-stats">
        <div className="result-stat-line">
          <span>Model confidence</span>
          <strong>{value.toFixed(1)}%</strong>
        </div>
        <div className="result-progress">
          <div style={{ width: `${value}%` }} />
        </div>
        <div className="result-stat-line result-stability">
          <span>Stability votes</span>
          <strong>
            {stableCount} / {requiredCount}
          </strong>
        </div>
        <div className="result-progress result-progress-alt">
          <div
            style={{
              width: `${Math.min(100, (stableCount / requiredCount) * 100)}%`,
            }}
          />
        </div>
      </div>
      <div className="result-note">
        <AlertCircle size={16} />
        <span>
          Confidence is the model's score, not a guarantee of correctness.
        </span>
      </div>
    </div>
  );
}
