import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { BookOpen, Plus, Search, Info } from "lucide-react";
import api from "../services/api";
import { useAuth } from "../context/AuthContext";
import "./AddSign.css";
export default function AddSign() {
  const { user } = useAuth();
  const [form, setForm] = useState({ name: "", description: "" }),
    [signs, setSigns] = useState([]),
    [query, setQuery] = useState(""),
    [loading, setLoading] = useState(false),
    [fetching, setFetching] = useState(true),
    [message, setMessage] = useState(""),
    [error, setError] = useState("");
  async function loadSigns() {
    setFetching(true);
    try {
      const { data } = await api.get("/sign");
      setSigns(Array.isArray(data.signs) ? data.signs : []);
    } catch (err) {
      setError(err.response?.data?.message || "Unable to load sign library");
    } finally {
      setFetching(false);
    }
  }
  useEffect(() => {
    loadSigns();
  }, []);
  async function submit(e) {
    e.preventDefault();
    if (!form.name.trim()) return;
    setLoading(true);
    setError("");
    setMessage("");
    try {
      await api.post("/sign", {
        name: form.name.trim(),
        description: form.description.trim(),
      });
      setForm({ name: "", description: "" });
      setMessage("Sign information saved successfully");
      await loadSigns();
    } catch (err) {
      setError(err.response?.data?.message || "Unable to save sign");
    } finally {
      setLoading(false);
    }
  }
  const filtered = signs.filter((s) =>
    `${s.name} ${s.description || ""}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  return (
    <div className="library-page">
      <div className="page-heading">
        <span className="eyebrow">SIGN MANAGEMENT</span>
        <h1>Sign Library</h1>
        <p>
          Organize descriptions and ideas for your recognition project. Adding a
          sign here does not retrain the AI model.
        </p>
      </div>
      <div className="library-layout">
        <section className="library-list panel">
          <div className="library-head">
            <h2>
              <BookOpen size={20} /> Saved signs
            </h2>
            <span>{signs.length} entries</span>
          </div>
          <div className="library-search">
            <Search size={18} />
            <input
              className="field"
              aria-label="Search signs"
              placeholder="Search by name or description"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          {fetching ? (
            <p className="library-empty">Loading signs...</p>
          ) : filtered.length ? (
            <div className="library-items">
              {filtered.map((sign) => (
                <div className="library-item" key={sign._id}>
                  <span className="library-item-mark">
                    {sign.name?.charAt(0)?.toUpperCase() || "?"}
                  </span>
                  <div>
                    <h3>{sign.name}</h3>
                    <p>{sign.description || "No description provided"}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="library-empty">No matching signs yet.</p>
          )}
        </section>
        <div className="library-side">
          <form className="library-form panel" onSubmit={submit}>
            <h2>
              <Plus size={20} /> Add sign information
            </h2>
            {user ? (
              <>
                <label htmlFor="sign-name">Sign name</label>
                <input
                  id="sign-name"
                  className="field"
                  required
                  maxLength={100}
                  placeholder="e.g. Hello"
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                />
                <label htmlFor="sign-description">Description</label>
                <textarea
                  id="sign-description"
                  className="field"
                  rows={5}
                  placeholder="Meaning or usage of this sign"
                  value={form.description}
                  onChange={(e) =>
                    setForm({ ...form, description: e.target.value })
                  }
                />
                {error && (
                  <div className="alert" role="alert">
                    {error}
                  </div>
                )}
                {message && (
                  <div className="alert alert-success" role="status">
                    {message}
                  </div>
                )}
                <button className="btn btn-primary" disabled={loading}>
                  <Plus size={17} />
                  {loading ? "Saving..." : "Save sign"}
                </button>
              </>
            ) : (
              <p className="library-empty">
                Please <Link to="/login">sign in</Link> to add sign information.
              </p>
            )}
          </form>
          <div className="library-tip panel">
            <Info size={19} />
            <p>
              <strong>Training note:</strong> The TensorFlow model recognizes
              only classes present in its training labels. New database entries
              are metadata, not new AI classes.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
