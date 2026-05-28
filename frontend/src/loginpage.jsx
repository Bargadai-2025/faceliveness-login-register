import React, { useState, useEffect } from "react";
import { getApiBase } from "./apiBase";
import "./loginpage.css";

function ShieldSVG({ large, keyhole }) {
  const size = large ? 64 : 32;
  if (keyhole) {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="#fff"
        width={size}
        height={size}
        style={{ display: "block" }}
        aria-hidden
      >
        <path
          fillRule="evenodd"
          d="M12.516 2.17a.75.75 0 0 0-1.032 0 11.209 11.209 0 0 1-7.877 3.08.75.75 0 0 0-.722.515A12.74 12.74 0 0 0 2.25 9.75c0 5.942 4.064 10.933 9.563 12.348a.749.749 0 0 0 .374 0c5.499-1.415 9.563-6.406 9.563-12.348 0-1.39-.223-2.73-.635-3.985a.75.75 0 0 0-.722-.516l-.143.001c-2.996 0-5.717-1.17-7.734-3.08Z"
          clipRule="evenodd"
        />
        <path
          fill="#0d3319"
          d="M12 11.25a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5Zm-1.5 2.25a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Z"
        />
        <path fill="#0d3319" d="M11.25 14.25v2.25h1.5v-2.25h-1.5Z" />
      </svg>
    );
  }
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="#fff"
      width={size}
      height={size}
      style={{ display: "block" }}
      aria-hidden
    >
      <path
        fillRule="evenodd"
        d="M12.516 2.17a.75.75 0 0 0-1.032 0 11.209 11.209 0 0 1-7.877 3.08.75.75 0 0 0-.722.515A12.74 12.74 0 0 0 2.25 9.75c0 5.942 4.064 10.933 9.563 12.348a.749.749 0 0 0 .374 0c5.499-1.415 9.563-6.406 9.563-12.348 0-1.39-.223-2.73-.635-3.985a.75.75 0 0 0-.722-.516l-.143.001c-2.996 0-5.717-1.17-7.734-3.08Zm3.094 8.016a.75.75 0 1 0-1.22-.872l-3.236 4.53L9.53 12.22a.75.75 0 0 0-1.06 1.06l2.25 2.25a.75.75 0 0 0 1.14-.094l3.75-5.25Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

function FingerprintIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="28" height="28" aria-hidden>
      <path
        fillRule="evenodd"
        d="M12 3.75a6.715 6.715 0 0 0-3.722 1.118.75.75 0 1 1-.828-1.25 8.25 8.25 0 0 1 12.8 6.883c0 3.014-.574 5.897-1.62 8.543a.75.75 0 0 1-1.395-.551A21.69 21.69 0 0 0 18.75 10.5 6.75 6.75 0 0 0 12 3.75ZM6.157 5.739a.75.75 0 0 1 .21 1.04A6.715 6.715 0 0 0 5.25 10.5c0 1.613-.463 3.12-1.265 4.393a.75.75 0 0 1-1.27-.8A6.715 6.715 0 0 0 3.75 10.5c0-1.68.503-3.246 1.367-4.55a.75.75 0 0 1 1.04-.211ZM12 7.5a3 3 0 0 0-3 3c0 3.1-1.176 5.927-3.105 8.056a.75.75 0 1 1-1.112-1.008A10.459 10.459 0 0 0 7.5 10.5a4.5 4.5 0 1 1 9 0c0 .547-.022 1.09-.067 1.626a.75.75 0 0 1-1.495-.123c.041-.495.062-.996.062-1.503a3 3 0 0 0-3-3Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

const STORAGE_KEY = "facematch_user";
const REMEMBER_EMAIL_KEY = "facematch_remember_email";

/**
 * @param {{ onLogin: (email: string, agentLabel: string) => void }} props
 */
export default function LoginPage({ onLogin }) {
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [remember, setRemember] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const API_URL = getApiBase();

  useEffect(() => {
    try {
      const saved = localStorage.getItem(REMEMBER_EMAIL_KEY);
      if (saved) {
        setEmail(saved);
        setRemember(true);
      }
    } catch {
      /* ignore */
    }
  }, []);

  const goRegister = (e) => {
    e.preventDefault();
    window.history.pushState({}, "", "/register");
    window.dispatchEvent(new PopStateEvent("popstate"));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    const emailNorm = email.trim().toLowerCase();
    if (!emailNorm || !emailNorm.includes("@")) {
      setError("Please enter a valid email address.");
      return;
    }

    const otpClean = otp.replace(/\D/g, "");
    if (otpClean.length !== 5) {
      setError("Please enter a 5-digit OTP.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(
        `${API_URL}/auth/check-email?email=${encodeURIComponent(emailNorm)}`,
      );
      const data = await res.json().catch(() => ({}));

      if (res.status === 404 || data.detail === "Not Found") {
        setError(
          "Login API not found. Restart the backend from the backend folder: uvicorn main:app --reload --host 127.0.0.1 --port 8000",
        );
        return;
      }

      if (!res.ok || !data.ok) {
        const msg = data.error || "This email is not registered. Please register first.";
        setError(msg);
        return;
      }

      if (remember) {
        localStorage.setItem(REMEMBER_EMAIL_KEY, emailNorm);
      } else {
        localStorage.removeItem(REMEMBER_EMAIL_KEY);
      }

      const faceLabel = data.face_label || emailNorm;
      const session = { email: emailNorm, agentLabel: faceLabel };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
      onLogin(emailNorm, faceLabel);
      window.history.pushState({}, "", "/");
      window.dispatchEvent(new PopStateEvent("popstate"));
    } catch {
      setError("Could not verify email. Is the API server running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="lp-root">
      <div className="lp-frame">
        <div className="lp-corner lp-corner-tl" aria-hidden />
        <div className="lp-corner lp-corner-tr" aria-hidden />
        <div className="lp-corner lp-corner-bl" aria-hidden />
        <div className="lp-corner lp-corner-br" aria-hidden />

        <div className="lp-shields" aria-hidden>
          <ShieldSVG large={false} />
          <ShieldSVG large keyhole />
          <ShieldSVG large={false} />
        </div>

        <div className="lp-card">
          <h1 className="lp-title">Face Match</h1>

          <form onSubmit={handleSubmit}>
            <div className="lp-field">
              <label className="lp-label" htmlFor="lp-email">
                User ID
              </label>
              <input
                id="lp-email"
                type="email"
                className="lp-input"
                placeholder="enter your email address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={loading}
                autoComplete="email"
              />
            </div>

            <div className="lp-field">
              <label className="lp-label" htmlFor="lp-otp">
                OTP
              </label>
              <input
                id="lp-otp"
                type="text"
                inputMode="numeric"
                autoComplete="one-time-code"
                className="lp-input lp-input-otp"
                placeholder="enter 5-digit OTP"
                maxLength={5}
                value={otp}
                onChange={(e) => setOtp(e.target.value.replace(/\D/g, "").slice(0, 5))}
                disabled={loading}
              />
            </div>

            <div className="lp-row-options">
              <label className="lp-remember">
                <input
                  type="checkbox"
                  checked={remember}
                  onChange={(e) => setRemember(e.target.checked)}
                  disabled={loading}
                />
                remember me
              </label>
              <span className="lp-forgot">forgot password?</span>
            </div>

            {error ? <div className="lp-error">{error}</div> : null}

            <button type="submit" className="lp-btn-login" disabled={loading}>
              {loading ? "Verifying..." : "LOGIN"}
            </button>
          </form>

          <p className="lp-help">
            Please contact administrator in case you are unable to login.
          </p>
          <p className="lp-help lp-help-register">
            New user?{" "}
            <a href="/register" onClick={goRegister} className="lp-link">
              Register here
            </a>
          </p>
        </div>

        <div className="lp-fingerprint" aria-hidden>
          <FingerprintIcon />
        </div>
      </div>
    </div>
  );
}

export { STORAGE_KEY };
