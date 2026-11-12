import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { LogIn, Mail, LockKeyhole } from "lucide-react";
import api from "../services/api";
import { useAuth } from "../context/AuthContext";
import "./Auth.css";
export default function Login() {
  const navigate = useNavigate(),
    { login } = useAuth();
  const [form, setForm] = useState({ email: "", password: "" }),
    [loading, setLoading] = useState(false),
    [error, setError] = useState("");
  async function submit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const { data } = await api.post("/auth/login", form);
      if (!data?.token || !data?.user)
        throw new Error("Server did not return user and token");
      login(data.user, data.token);
      navigate("/live", { replace: true });
    } catch (err) {
      setError(err.response?.data?.message || err.message || "Login failed");
    } finally {
      setLoading(false);
    }
  }
  return (
    <div className="auth-page">
      <div className="auth-intro">
        <span className="eyebrow">WELCOME BACK</span>
        <h1>Sign in to SignAI</h1>
        <p>Access your recognition workspace and saved predictions.</p>
      </div>
      <form className="auth-form panel" onSubmit={submit}>
        <label htmlFor="login-email">Email address</label>
        <div className="input-icon">
          <Mail size={18} />
          <input
            id="login-email"
            type="email"
            autoComplete="email"
            required
            className="field"
            placeholder="you@example.com"
            value={form.email}
            onChange={(e) => setForm({ ...form, email: e.target.value })}
          />
        </div>
        <label htmlFor="login-password">Password</label>
        <div className="input-icon">
          <LockKeyhole size={18} />
          <input
            id="login-password"
            type="password"
            autoComplete="current-password"
            required
            className="field"
            placeholder="Enter your password"
            value={form.password}
            onChange={(e) => setForm({ ...form, password: e.target.value })}
          />
        </div>
        {error && (
          <div className="alert" role="alert">
            {error}
          </div>
        )}
        <button className="btn btn-primary" disabled={loading}>
          <LogIn size={17} />
          {loading ? "Signing in..." : "Sign in"}
        </button>
        <p className="auth-switch">
          New to SignAI? <Link to="/register">Create an account</Link>
        </p>
      </form>
    </div>
  );
}
