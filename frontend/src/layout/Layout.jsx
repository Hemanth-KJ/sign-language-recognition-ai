import { Outlet } from "react-router-dom";
import Navbar from "../components/Navbar";
import "./Layout.css";
export default function Layout() {
  return (
    <div className="app-shell">
      <Navbar />
      <main className="main-content">
        <Outlet />
      </main>
      <footer className="site-footer">
        <div>
          <strong>
            Sign<span>AI</span>
          </strong>
          <p>Computer vision for accessible communication.</p>
        </div>
        <small>React · Express · FastAPI · TensorFlow · MediaPipe</small>
      </footer>
    </div>
  );
}
