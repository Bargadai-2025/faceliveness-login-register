import { useState, useEffect } from "react";
import FaceMatch from "./FaceMatch";
import FaceRegister from "./FaceRegister";
import LoginPage from "./loginpage";

function App() {
  // Default user to bypass login
  const [user, setUser] = useState({ 
    email: "agent@bargad.ai", 
    agentLabel: "Authorized Agent" 
  });
  const [currentPath, setCurrentPath] = useState(window.location.pathname);

  useEffect(() => {
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
    // Removed localStorage.setItem as per user request
  };

  const handleLogout = () => {
    setUser(null);
    // Removed localStorage.removeItem as per user request
  };

  // Routing logic
  if (currentPath === "/login") {
    return <LoginPage onLogin={handleLogin} />;
  }

  if (currentPath === "/register") {
    return (
      <FaceRegister
        userEmail={user?.email}
        userAgentLabel={user?.agentLabel}
        onLogout={handleLogout}
      />
    );
  }

  // Default to FaceMatch
  return (
    <FaceMatch
      userEmail={user?.email}
      userAgentLabel={user?.agentLabel}
      onLogout={handleLogout}
    />
  );
}

export default App;

