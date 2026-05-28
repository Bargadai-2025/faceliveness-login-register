import { useState, useEffect } from "react";
import FaceMatch from "./FaceMatch";
import FaceRegister from "./FaceRegister";
import LoginPage, { STORAGE_KEY } from "./loginpage";

function readStoredUser() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed?.email && parsed?.agentLabel) return parsed;
    return null;
  } catch {
    return null;
  }
}

function App() {
  const [user, setUser] = useState(() => readStoredUser());
  const [currentPath, setCurrentPath] = useState(window.location.pathname);

  useEffect(() => {
    const handleLocationChange = () => {
      setCurrentPath(window.location.pathname);
    };
    window.addEventListener("popstate", handleLocationChange);
    return () => window.removeEventListener("popstate", handleLocationChange);
  }, []);

  useEffect(() => {
    if (user && currentPath === "/login") {
      window.history.replaceState({}, "", "/");
      setCurrentPath("/");
    }
  }, [user, currentPath]);

  const handleLogin = (email, agentLabel) => {
    const next = { email, agentLabel };
    setUser(next);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem(STORAGE_KEY);
    window.history.pushState({}, "", "/login");
    window.dispatchEvent(new PopStateEvent("popstate"));
  };

  if (currentPath === "/register") {
    return (
      <FaceRegister
        userEmail={user?.email}
        userAgentLabel={user?.agentLabel}
        onLogout={handleLogout}
        onRegistered={(email, faceLabel) => handleLogin(email, faceLabel)}
      />
    );
  }

  if (!user) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return (
    <FaceMatch
      userEmail={user.email}
      userAgentLabel={user.agentLabel}
      onLogout={handleLogout}
    />
  );
}

export default App;
