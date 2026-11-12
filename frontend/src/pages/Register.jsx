import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { UserRound, Mail, LockKeyhole, UserPlus } from "lucide-react";
import api from "../services/api";
import { useAuth } from "../context/AuthContext";
import "./Auth.css";
export default function Register() {
  const navigate = useNavigate(),
    { login } = useAuth();
  const [form, setForm] = useState({ name: "", email: "", password: "" }),
    [loading, setLoading] = useState(false),
    [error, setError] = useState("");
  async function submit(e) {
    e.preventDefault();
    setError("");
    if (form.password.length < 8) {
      setError("Password must contain at least 8 characters");
      return;
    }
    setLoading(true);
    try {
      const { data } = await api.post("/auth/register", form);
      if (!data?.token || !data?.user)
        throw new Error("Server did not return user and token");
      login(data.user, data.token);
      navigate("/live", { replace: true });
    } catch (err) {
      setError(
        err.response?.data?.message || err.message || "Registration failed",
      );
    } finally {
      setLoading(false);
    }
  }
  return (
    <div className="auth-page">
      <div className="auth-intro">
        <span className="eyebrow">GET STARTED</span>
        <h1>Create your account</h1>
        <p>Join SignAI to manage signs and explore AI-powered recognition.</p>
      </div>
      <form className="auth-form panel" onSubmit={submit}>
        <label htmlFor="reg-name">Full name</label>
        <div className="input-icon">
          <UserRound size={18} />
          <input
            id="reg-name"
            required
            className="field"
            autoComplete="name"
            maxLength={100}
            placeholder="Your name"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
          />
        </div>
        <label htmlFor="reg-email">Email address</label>
        <div className="input-icon">
          <Mail size={18} />
          <input
            id="reg-email"
            type="email"
            required
            autoComplete="email"
            className="field"
            placeholder="you@example.com"
            value={form.email}
            onChange={(e) => setForm({ ...form, email: e.target.value })}
          />
        </div>
        <label htmlFor="reg-password">Password</label>
        <div className="input-icon">
          <LockKeyhole size={18} />
          <input
            id="reg-password"
            type="password"
            required
            minLength={8}
            autoComplete="new-password"
            className="field"
            placeholder="At least 8 characters"
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
          <UserPlus size={17} />
          {loading ? "Creating account..." : "Create account"}
        </button>
        <p className="auth-switch">
          Already have an account? <Link to="/login">Sign in</Link>
        </p>
      </form>
    </div>
  );
}
