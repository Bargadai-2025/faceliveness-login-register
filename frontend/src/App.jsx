import { useState, useEffect } from "react";
import FaceMatch from "./FaceMatch";
import FaceRegister from "./FaceRegister";
import LoginPage from "./loginpage";

const AUTH_STORAGE_KEY = "facematch_auth";

function readStoredUser() {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (data && typeof data.email === "string") {
      return { 
        email: data.email,
        agentLabel: data.agentLabel || null
      };
    }
  } catch {
    /* ignore */
  }
  return null;  
}

function App() {
  const [user, setUser] = useState(null);
  const [currentPath, setCurrentPath] = useState(window.location.pathname);

  useEffect(() => {
    setUser(readStoredUser());
    
    // Listen for path changes
    const handleLocationChange = () => {
      setCurrentPath(window.location.pathname);
    };
    window.addEventListener("popstate", handleLocationChange);
    return () => window.removeEventListener("popstate", handleLocationChange);
  }, []);

  const handleLogin = (email, agentLabel) => {
    const next = { email, agentLabel };
    setUser(next);
    localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(next));
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem(AUTH_STORAGE_KEY);
  };

  if (!user) {
    return <LoginPage onLogin={handleLogin} />;
  }

  if (currentPath === "/register") {
    return <FaceRegister userEmail={user.email} userAgentLabel={user.agentLabel} onLogout={handleLogout} />;
  }

  return <FaceMatch userEmail={user.email} userAgentLabel={user.agentLabel} onLogout={handleLogout} />;
}

export default App;

