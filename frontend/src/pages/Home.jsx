import { Link } from "react-router-dom";
import {
  ArrowUpRight,
  Camera,
  BrainCircuit,
  Database,
  ShieldCheck,
  ScanLine,
  Hand,
  Activity,
} from "lucide-react";
import "./Home.css";
const features = [
  {
    icon: Camera,
    title: "Live camera input",
    text: "Capture hand gestures directly from your webcam.",
  },
  {
    icon: ScanLine,
    title: "Landmark extraction",
    text: "MediaPipe detects 21 landmarks per hand, normalized for inference.",
  },
  {
    icon: BrainCircuit,
    title: "TensorFlow inference",
    text: "A trained 126-feature classifier estimates the displayed sign.",
  },
  {
    icon: Database,
    title: "Sign management",
    text: "Manage sign descriptions and review recognized signs.",
  },
];
export default function Home() {
  return (
    <div className="home">
      <section className="home-hero">
        <div className="hero-copy">
          <div className="eyebrow">
            <span className="hero-dot" /> FINAL YEAR PROJECT · COMPUTER VISION
          </div>
          <h1>
            Make every <em>gesture</em> count.
          </h1>
          <p>
            SignAI turns camera-captured hand poses into real-time sign
            predictions using MediaPipe landmarks and a TensorFlow classifier.
          </p>
          <div className="hero-actions">
            <Link className="btn btn-primary" to="/live">
              <Camera size={18} /> Launch Live AI <ArrowUpRight size={17} />
            </Link>
            <Link className="btn btn-outline" to="/add-sign">
              Explore Sign Library
            </Link>
          </div>
          <div className="hero-points">
            <span>
              <ShieldCheck size={17} /> Local model inference
            </span>
            <span>
              <Activity size={17} /> Live confidence feedback
            </span>
          </div>
        </div>
        <div className="hero-visual panel">
          <div className="visual-top">
            <span>
              <span className="live-dot" /> LIVE RECOGNITION
            </span>
            <span>MODEL · HAND-126</span>
          </div>
          <div className="visual-center">
            <div className="hand-halo">
              <Hand size={112} strokeWidth={1.1} />
            </div>
            <div className="visual-letter">
              A<span>Example prediction</span>
            </div>
          </div>
          <div className="visual-bottom">
            <span>Gesture recognition</span>
            <strong>TensorFlow + MediaPipe</strong>
          </div>
        </div>
      </section>
      <section className="home-metrics">
        <div>
          <strong>126</strong>
          <span>Hand landmark features</span>
        </div>
        <div>
          <strong>28</strong>
          <span>Expected trained classes</span>
        </div>
        <div>
          <strong>2</strong>
          <span>Supported hands</span>
        </div>
        <div>
          <strong>Live</strong>
          <span>Webcam recognition</span>
        </div>
      </section>
      <section className="home-section">
        <div className="section-heading">
          <div>
            <span className="eyebrow">CAPABILITIES</span>
            <h2>Everything in one recognition workspace</h2>
          </div>
          <p>
            From capturing a hand pose to understanding the model's response.
          </p>
        </div>
        <div className="feature-grid">
          {features.map(({ icon: Icon, title, text }) => (
            <div className="feature-panel panel" key={title}>
              <div className="feature-icon">
                <Icon size={23} />
              </div>
              <h3>{title}</h3>
              <p>{text}</p>
            </div>
          ))}
        </div>
      </section>
      <section className="home-banner panel">
        <div>
          <span className="eyebrow">READY TO TEST?</span>
          <h2>Start recognizing signs with your webcam.</h2>
          <p>
            Use a well-lit environment, keep your hand visible, and hold each
            pose steady.
          </p>
        </div>
        <Link to="/live" className="btn btn-primary">
          Open Recognition Studio <ArrowUpRight size={18} />
        </Link>
      </section>
      <p className="home-disclaimer">
        Research prototype: recognition is limited by the trained dataset.
        Motion-dependent signs and continuous signing are not yet supported.
      </p>
    </div>
  );
}
