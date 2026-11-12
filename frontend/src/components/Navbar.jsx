import { useState } from "react";
import { NavLink, Link, useNavigate } from "react-router-dom";
import { Menu, X, Hand, LogOut } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import "./Navbar.css";
export default function Navbar() {
  const [open, setOpen] = useState(false);
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const cls = ({ isActive }) => `site-nav-link ${isActive ? "selected" : ""}`;
  const signOut = () => {
    logout();
    setOpen(false);
    navigate("/");
  };
  return (
    <header className="site-header">
      <div className="site-nav">
        <Link to="/" className="site-logo" onClick={() => setOpen(false)}>
          <span className="logo-mark">
            <Hand size={22} />
          </span>
          <span>
            Sign<span className="logo-accent">AI</span>
            <small>INTELLIGENT RECOGNITION</small>
          </span>
        </Link>
        <nav
          className={`site-links ${open ? "is-open" : ""}`}
          aria-label="Main navigation"
        >
          <NavLink to="/" end className={cls} onClick={() => setOpen(false)}>
            Overview
          </NavLink>
          <NavLink to="/live" className={cls} onClick={() => setOpen(false)}>
            Live Recognition
          </NavLink>
          <NavLink
            to="/add-sign"
            className={cls}
            onClick={() => setOpen(false)}
          >
            Sign Library
          </NavLink>
          <div className="mobile-account">
            {user ? (
              <button onClick={signOut} className="btn btn-outline">
                <LogOut size={16} /> Log out
              </button>
            ) : (
              <>
                <Link
                  onClick={() => setOpen(false)}
                  to="/login"
                  className="btn btn-outline"
                >
                  Log in
                </Link>
                <Link
                  onClick={() => setOpen(false)}
                  to="/register"
                  className="btn btn-primary"
                >
                  Get started
                </Link>
              </>
            )}
          </div>
        </nav>
        <div className="desktop-account">
          {user ? (
            <>
              <span className="account-name">
                <span className="account-avatar">
                  {user.name?.charAt(0)?.toUpperCase() || "U"}
                </span>
                {user.name}
              </span>
              <button className="btn btn-outline" onClick={signOut}>
                <LogOut size={15} /> Logout
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="btn btn-outline">
                Log in
              </Link>
              <Link to="/register" className="btn btn-primary">
                Get started
              </Link>
            </>
          )}
        </div>
        <button
          className="menu-toggle"
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? <X /> : <Menu />}
        </button>
      </div>
    </header>
  );
}
