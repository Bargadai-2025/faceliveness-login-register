import React, { useState } from "react";
import { getApiBase } from "./apiBase";
import "./loginpage.css";

function ShieldSVG({ large }) {
  const size = large ? 64 : 32;
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="#fff"
      width={size}
      height={size}
      style={{ display: "block" }}
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
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="28" height="28">
      <path
        fillRule="evenodd"
        d="M12 3.75a6.715 6.715 0 0 0-3.722 1.118.75.75 0 1 1-.828-1.25 8.25 8.25 0 0 1 12.8 6.883c0 3.014-.574 5.897-1.62 8.543a.75.75 0 0 1-1.395-.551A21.69 21.69 0 0 0 18.75 10.5 6.75 6.75 0 0 0 12 3.75ZM6.157 5.739a.75.75 0 0 1 .21 1.04A6.715 6.715 0 0 0 5.25 10.5c0 1.613-.463 3.12-1.265 4.393a.75.75 0 0 1-1.27-.8A6.715 6.715 0 0 0 3.75 10.5c0-1.68.503-3.246 1.367-4.55a.75.75 0 0 1 1.04-.211ZM12 7.5a3 3 0 0 0-3 3c0 3.1-1.176 5.927-3.105 8.056a.75.75 0 1 1-1.112-1.008A10.459 10.459 0 0 0 7.5 10.5a4.5 4.5 0 1 1 9 0c0 .547-.022 1.09-.067 1.626a.75.75 0 0 1-1.495-.123c.041-.495.062-.996.062-1.503a3 3 0 0 0-3-3Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

/**
 * @param {{ onLogin: (email: string, agentLabel: string) => void }} props
 */
/**
 * @param {{ onLogin: (email: string, agentLabel: string) => void }} props
 */
export default function LoginPage({ onLogin }) {
  const [error, setError] = useState("");

  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState("");
  const [loadingAgents, setLoadingAgents] = useState(true);

  const API_URL = getApiBase();

  React.useEffect(() => {
    async function fetchAgents() {
      try {
        const res = await fetch(`${API_URL}/agents/list`);
        if (!res.ok) throw new Error("Failed to load agents");
        const data = await res.json();
        setAgents(data);
        if (data.length > 0) setSelectedAgent(data[0].label);
      } catch (e) {
        console.error("Agent fetch error:", e);
      } finally {
        setLoadingAgents(false);
      }
    }
    fetchAgents();
  }, [API_URL]);

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");
    if (!selectedAgent) {
      setError("Please select an agent to proceed.");
      return;
    }
    // Log in using the agent's label as their identity
    onLogin(selectedAgent, selectedAgent);
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
          <ShieldSVG large />
          <ShieldSVG large={false} />
        </div>

        <div className="lp-card">
          <h2 className="lp-title">Agent Portal</h2>

          <form onSubmit={handleSubmit}>
            <div className="lp-field">
              <label className="lp-label" htmlFor="lp-agent">
                Select Identity <span className="lp-badge">Authorized</span>
              </label>
              <div className="lp-select-wrap">
                <select
                  id="lp-agent"
                  className="lp-input lp-select"
                  value={selectedAgent}
                  onChange={(e) => setSelectedAgent(e.target.value)}
                  disabled={loadingAgents}
                >
                  {loadingAgents ? (
                    <option>Loading authorized agents...</option>
                  ) : (
                    agents.map((a) => (
                      <option key={a.label} value={a.label} style={{ color: "#000" }}>
                        {a.label}
                      </option>
                    ))
                  )}
                </select>
                <div className="lp-select-icon">▼</div>
              </div>
            </div>

            <div className="lp-row-options">
              <label className="lp-remember">
                <input type="checkbox" checked readOnly style={{ accentColor: "#24aa4d" }} />
                Secure Session
              </label>
              <span style={{ opacity: 0.6 }}>Auth-V2</span>
            </div>

            {error ? <div className="lp-error">{error}</div> : null}

            <button type="submit" className="lp-btn-login" disabled={loadingAgents}>
              Proceed to Dashboard
            </button>
          </form>

          <p className="lp-help">
            Select your assigned agent ID to begin liveness verification.
          </p>
        </div>

        <div className="lp-fingerprint" aria-hidden>
          <FingerprintIcon />
        </div>
      </div>
    </div>
  );
}
