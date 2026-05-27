import re
import sys

def main():
    file_path = r'c:\Users\yash jadhav\Desktop\cursor_facematch\faceliveness-update\frontend\src\FaceMatch.jsx'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find where return ( starts
    match = re.search(r'  return \(\n    <div className="fm-page">', content)
    if not match:
        print("Could not find the return statement")
        return

    return_start_idx = match.start()

    # Add Power to lucide-react imports if not there
    if 'Power' not in content:
        content = content.replace('Loader2,', 'Loader2, Power, CheckCircle, Play,')

    new_return = """  return (
    <div className="fm-page">
      {/* Toast Notification Pipeline */}
      <div className="fm-toast-pipeline">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`fm-floating-toast ${t.type || "success"}`}
          >
            <div className="fm-toast-icon">
              {t.type === "warning" ? (
                <AlertTriangle size={18} color="#ffbf01" />
              ) : t.type === "error" ? (
                <AlertOctagon size={18} color="#ff4444" />
              ) : (
                <UserCheck size={18} color="#24aa4d" />
              )}
            </div>
            <span>{t.msg}</span>
          </div>
        ))}
      </div>

      <header className="fm-header-banner">
        <div className="fm-header-left">
          <div className="fm-header-text">
            <span className="fm-demo-text">DEMO</span>
            <h1>FACE BIOMETRICS</h1>
            <p>FACE MATCH, LIVELINESS, DEEP FAKE & LOCATION</p>
          </div>
        </div>
        <div className="fm-header-right">
          <Power className="fm-power-btn" onClick={onLogout} size={28} />
        </div>
      </header>

      <div className={`fm-main-layout ${showCamera ? "fm-camera-active" : ""}`}>
        <div className="fm-left-col">
          <div className="fm-camera-outer">
            <div className="fm-camera-container">
              <div className="fm-main-camera-contianer-relative">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="fm-camera-feed"
                />
                <canvas
                  ref={overlayCanvasRef}
                  className="fm-mesh-overlay"
                />
                <canvas ref={canvasRef} style={{ display: "none" }} />
                
                {/* High-tech Viewfinder Corners */}
                <div className="fm-viewfinder-corner top-left"></div>
                <div className="fm-viewfinder-corner top-right"></div>
                <div className="fm-viewfinder-corner bottom-left"></div>
                <div className="fm-viewfinder-corner bottom-right"></div>

                {!showCamera && !loading && (
                    <div className="fm-start-overlay">
                        <button className="fm-start-btn" onClick={startCamera}>
                            START <Play size={20} fill="currentColor" />
                        </button>
                    </div>
                )}
                
                {showCamera && (
                    <div className="fm-scanline"></div>
                )}

                {/* Remove the old error overlay here because it goes in map now */}
                
                <div className="fm-camera-actions">
                  {(livenessLive || livenessStep === "complete" || livenessStep === "capture") && !multiPersonError && (
                    <button className="fm-capture-btn" onClick={takeSelfie}>
                      <Camera size={18} /> Capture Selfie
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
          
          <div className="fm-challenges-pills">
             {[1, 2, 3].map(num => {
                 const isActive = challengeIndex + 1 === num && livenessStep === "gesture";
                 const isCompleted = challengeIndex >= num || livenessLive;
                 return (
                     <div key={num} className={`fm-pill ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}>
                         Challenge {num} <CheckCircle size={14} className="fm-pill-icon" />
                     </div>
                 )
             })}
             <div className={`fm-pill ${livenessLive ? 'active' : ''}`}>
                 Capture Selfie <Camera size={14} className="fm-pill-icon" />
             </div>
          </div>
        </div>

        <div className="fm-right-col">
          <div className={`fm-geo-card ${error ? 'fm-geo-error-active' : ''}`}>
            {error && (
                <div className="fm-map-error-overlay">
                   <AlertTriangle size={20} /> {error}
                </div>
            )}
            <div className="fm-geo-map">
              {geoData ? (
                <MapContainer
                  center={[parseFloat(geoData.lat), parseFloat(geoData.long)]}
                  zoom={16}
                  style={{ width: "100%", height: "100%" }}
                  zoomControl={false}
                  dragging={false}
                  scrollWheelZoom={false}
                  doubleClickZoom={false}
                  attributionControl={false}
                >
                  <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                  <Marker position={[parseFloat(geoData.lat), parseFloat(geoData.long)]} />
                </MapContainer>
              ) : (
                <div className="fm-map-placeholder">
                  <MapPin size={24} />
                  <span>Map Ready</span>
                </div>
              )}
            </div>
            <div className="fm-geo-details">
              <div className="fm-geo-full-address">
                 {geoAddress ? geoAddress.full : "Fetching location..."}
              </div>
              <div className="fm-geo-coords-row">
                 <div>
                    <span className="geo-label">Latitude</span> {geoData ? parseFloat(geoData.lat).toFixed(5) : "0.00000"}° N
                 </div>
                 <div>
                    <span className="geo-label">Longitude</span> {geoData ? parseFloat(geoData.long).toFixed(5) : "0.00000"}° E
                 </div>
                 <div className="geo-time">
                    {geoData ? new Date(geoData.timestamp).toLocaleString() : "Date / Time"}
                 </div>
              </div>
            </div>
          </div>

          <div className="fm-matches-container">
            <div className="fm-match-images">
              <div className="fm-match-box">
                <div className="fm-match-label">#1 MATCHED (DB)</div>
                {results.length > 0 && results[0].label !== "txt" ? (
                  <img src={results[0].matched_image || (results[0].images && results[0].images[0])} alt="DB" />
                ) : (
                  <div className="fm-img-placeholder"></div>
                )}
              </div>
              <div className="fm-match-box">
                <div className="fm-match-label">CAPTURED (LIVE)</div>
                {capturedImage || preview ? (
                  <img src={capturedImage || preview} alt="Live" />
                ) : (
                  <div className="fm-img-placeholder"></div>
                )}
              </div>
            </div>
            <div className="fm-score-container">
              <div className="fm-score-header">
                <span>FACE MATCH SCORE</span> <Info size={14} className="fm-info-icon" />
              </div>
              <div className="fm-slider-track">
                <span className="fm-slider-label">Low</span>
                <div className="fm-slider-bar">
                  <div className="fm-slider-fill" style={{ width: (results.length > 0 ? (results[0].confidence * 100) : 0) + '%' }}></div>
                  <div className="fm-slider-thumb-wrapper" style={{ left: (results.length > 0 ? (results[0].confidence * 100) : 0) + '%' }}>
                    <div className="fm-slider-thumb">
                       {results.length > 0 ? Math.round(results[0].confidence * 100) : 0}%
                    </div>
                  </div>
                </div>
                <span className="fm-slider-label">High</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="fm-footer-branding" style={{ bottom: 10, right: 20 }}>
        <img src={bargadBranding} alt="Bargad" style={{width: '120px'}}/>
      </div>
    </div>
  );
}
"""

    final_content = content[:return_start_idx] + new_return
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(final_content)

if __name__ == "__main__":
    main()
