import React, { useState, useEffect, useCallback, useRef } from "react";
import { Play, Pause, Upload, Compass } from "lucide-react";

const API_BASE = window.location.origin;

export default function App() {
  // Metadata & Timestep state
  const [numFrames, setNumFrames] = useState<number>(0);
  const [, setAllSteps] = useState<number[]>([]);
  const [timestep, setTimestep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackFps, setPlaybackFps] = useState<number>(15);

  // Pan & Zoom State
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0 });

  // Settings & Toggles
  const [visThreshold, setVisThreshold] = useState<number>(3.0);
  const [bevStride] = useState<number>(3);
  const [status, setStatus] = useState<string>("Ready");

  // Map Alignment State
  const [alignment, setAlignment] = useState({
    offset_x: 0.0,
    offset_z: 0.0,
    offset_y: 0.0,
    scale: 1.0,
    rotation: 0.0,
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch initial metadata and alignment
  const fetchMetadata = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/metadata`);
      const data = await res.json();
      setNumFrames(data.num_frames);
      setAllSteps(data.all_steps);

      // Fetch alignment
      const alignRes = await fetch(`${API_BASE}/api/alignment`);
      const alignData = await alignRes.json();
      setAlignment(alignData);
    } catch (e) {
      console.error("Failed to fetch metadata:", e);
      setStatus("Error connecting to server");
    }
  }, []);

  useEffect(() => {
    fetchMetadata();
  }, [fetchMetadata]);

  const isImageLoading = useRef(false);

  // Playback timer loop (Smart debounced)
  useEffect(() => {
    if (!isPlaying || numFrames === 0) return;

    const interval = setInterval(() => {
      // Smart Sync: Wait for the previous map image to finish downloading before advancing!
      if (isImageLoading.current) {
        return; 
      }

      setTimestep((prev) => {
        const next = prev + 1;
        if (next >= numFrames) {
          setIsPlaying(false);
          return prev; // Stop at the end frame
        }
        isImageLoading.current = true; // Lock the next frame until it loads
        return next;
      });
    }, 1000 / playbackFps);

    return () => clearInterval(interval);
  }, [isPlaying, playbackFps, numFrames]);

  // Handle alignment updates to backend
  const updateAlignment = async (updated: typeof alignment) => {
    setAlignment(updated);
    const form = new FormData();
    form.append("offset_x", updated.offset_x.toString());
    form.append("offset_z", updated.offset_z.toString());
    form.append("offset_y", updated.offset_y.toString());
    form.append("scale", updated.scale.toString());
    form.append("rotation", updated.rotation.toString());
    
    try {
      await fetch(`${API_BASE}/api/alignment`, {
        method: "POST",
        body: form,
      });
    } catch (e) {
      console.error("Failed to save alignment settings:", e);
    }
  };

  // Handle threshold/sparsity updates
  const handleSettingsUpdate = async () => {
    const form = new FormData();
    form.append("vis_threshold", visThreshold.toString());
    form.append("bev_stride", bevStride.toString());
    try {
      await fetch(`${API_BASE}/api/settings`, {
        method: "POST",
        body: form,
      });
      fetchMetadata();
    } catch (e) {
      console.error("Failed to save settings:", e);
    }
  };

  // Video Upload handler - posts actual video file to backend
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setStatus("Uploading & Processing video (Running model inference)...");
    
    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: form,
      });
      if (res.ok) {
        setStatus("Processing successful!");
        setTimeout(() => setStatus("Ready"), 3000);
        // Refresh metadata and reset timeline
        await fetchMetadata();
        setTimestep(0);
        setIsPlaying(true); // Automatically start playback!
      } else {
        const errData = await res.json();
        setStatus(`Inference Error: ${errData.detail || "Unknown error"}`);
      }
    } catch (err) {
      console.error(err);
      setStatus("Network error during upload");
    }
  };

  // Trigger download of the current 2D map
  const handleDownloadMap = () => {
    const url = `${API_BASE}/api/2dmap/${timestep}`;
    const a = document.createElement("a");
    a.href = url;
    a.download = `lingbot_sparse_map_frame_${timestep}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="dashboard-container" style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Header */}
      <header className="dashboard-header" style={{ padding: "16px 24px", display: "flex", justifyContent: "space-between", alignItems: "center", backgroundColor: "#1e2128", borderBottom: "1px solid #2d313a" }}>
        <div className="header-brand" style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <Compass className="header-logo" size={32} style={{ color: "#38bdf8" }} />
          <div className="header-title">
            <h1 style={{ margin: 0, fontSize: "1.25rem", color: "#f8fafc" }}>LingBot-MAP Dashboard</h1>
            <p style={{ margin: 0, fontSize: "0.85rem", color: "#94a3b8" }}>Real-Time 2D Mapping & Localization</p>
          </div>
        </div>
        <div className="header-actions" style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          {numFrames > 0 && (
            <button onClick={handleDownloadMap} style={{ padding: "8px 16px", backgroundColor: "#38bdf8", color: "#0f172a", border: "none", borderRadius: "6px", fontWeight: 600, cursor: "pointer", fontSize: "0.85rem" }}>
              Download Map
            </button>
          )}
          <div className="status-badge" style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.875rem", color: "#94a3b8" }}>
            <span className="status-dot" style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: status === "Ready" ? "#10b981" : "#f59e0b" }}></span>
            Server Status: <span style={{ fontWeight: 600, color: "#e2e8f0" }}>{status}</span>
          </div>
        </div>
      </header>

      {/* Main Body */}
      <div className="dashboard-body" style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Left Sidebar Controls */}
        <aside className="sidebar-controls" style={{ width: "320px", backgroundColor: "#15181e", borderRight: "1px solid #2d313a", padding: "20px", display: "flex", flexDirection: "column", gap: "24px", overflowY: "auto" }}>
          
          <div className="control-group">
            <h3 style={{ fontSize: "0.95rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "#94a3b8", marginBottom: "12px" }}>Input & Playback</h3>
            <input ref={fileInputRef} type="file" accept="video/*" style={{ display: "none" }} onChange={handleUpload} />
            <button onClick={() => fileInputRef.current?.click()} className="custom-button" style={{ width: "100%", padding: "10px", display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", backgroundColor: "#38bdf8", color: "#0f172a", border: "none", borderRadius: "6px", fontWeight: 600, cursor: "pointer" }}>
              <Upload size={18} /> Upload Video
            </button>
            
            <div style={{ marginTop: "16px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", color: "#cbd5e1", marginBottom: "6px" }}>
                <span>Timeline</span>
                <span>Step: {timestep} / {Math.max(0, numFrames - 1)}</span>
              </div>
              <input 
                type="range" 
                min={0} 
                max={Math.max(0, numFrames - 1)} 
                value={timestep} 
                onChange={(e) => setTimestep(parseInt(e.target.value))}
                disabled={numFrames <= 1}
                style={{ width: "100%", accentColor: "#38bdf8" }}
              />
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: "12px" }}>
                <button onClick={() => setIsPlaying(!isPlaying)} style={{ background: "none", border: "none", cursor: "pointer", display: "flex", alignItems: "center", gap: "6px", color: "#f8fafc", fontSize: "0.9rem" }}>
                  {isPlaying ? <Pause size={20} style={{ color: "#f59e0b" }} /> : <Play size={20} style={{ color: "#10b981" }} />}
                  {isPlaying ? "Pause" : "Play"}
                </button>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ fontSize: "0.85rem", color: "#94a3b8" }}>FPS: {playbackFps}</span>
                  <input 
                    type="range" 
                    min={1} 
                    max={30} 
                    value={playbackFps} 
                    onChange={(e) => setPlaybackFps(parseInt(e.target.value))}
                    style={{ width: "80px", accentColor: "#10b981" }}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="control-group">
            <h3 style={{ fontSize: "0.95rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "#94a3b8", marginBottom: "12px" }}>Map Settings</h3>
            <div style={{ marginBottom: "12px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", color: "#cbd5e1", marginBottom: "4px" }}>
                <span>Confidence Threshold</span>
                <span>{visThreshold}</span>
              </div>
              <input 
                type="range" 
                min={0} 
                max={10} 
                step={0.1}
                value={visThreshold} 
                onChange={(e) => setVisThreshold(parseFloat(e.target.value))}
                onMouseUp={handleSettingsUpdate}
                style={{ width: "100%", accentColor: "#a855f7" }}
              />
            </div>
          </div>

          <div className="control-group">
            <h3 style={{ fontSize: "0.95rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "#94a3b8", marginBottom: "12px" }}>Interactive Alignment</h3>
            
            {[{ label: "Offset X (Horizontal)", key: "offset_x", min: -20, max: 20, step: 0.1 },
              { label: "Offset Z (Depth)", key: "offset_z", min: -20, max: 20, step: 0.1 },
              { label: "Height Y", key: "offset_y", min: -10, max: 10, step: 0.05 },
              { label: "Scale", key: "scale", min: 0.1, max: 5, step: 0.05 },
              { label: "Rotation (Yaw)", key: "rotation", min: 0, max: 360, step: 1 }].map((field) => (
              <div key={field.key} style={{ marginBottom: "12px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", color: "#cbd5e1", marginBottom: "4px" }}>
                  <span>{field.label}</span>
                  <span>{(alignment as any)[field.key]}{field.key === 'rotation' ? '°' : ''}</span>
                </div>
                <input 
                  type="range" 
                  min={field.min} 
                  max={field.max} 
                  step={field.step}
                  value={(alignment as any)[field.key]} 
                  onChange={(e) => updateAlignment({ ...alignment, [field.key]: parseFloat(e.target.value) })}
                  style={{ width: "100%", accentColor: "#f43f5e" }}
                />
              </div>
            ))}
          </div>

        </aside>

        {/* Center Main Viewport (Video + Map) */}
        <main className="scene-viewport" style={{ flex: 1, backgroundColor: "#0f1115", position: "relative", overflow: "hidden" }}>
          
          {/* 2D Map Area (Fills entire main viewport) */}
          <div 
            style={{ 
              width: "100%", 
              height: "100%", 
              backgroundColor: "#1e2128", 
              cursor: "grab", 
              position: "relative", 
              overflow: "hidden" 
            }}
            onWheel={(e) => setZoom(z => Math.min(Math.max(0.1, z - e.deltaY * 0.002), 10))}
            onMouseDown={(e) => { 
              isDragging.current = true; 
              dragStart.current = { x: e.clientX - pan.x, y: e.clientY - pan.y }; 
              (e.target as HTMLElement).style.cursor = 'grabbing';
            }}
            onMouseMove={(e) => { 
              if (isDragging.current) setPan({ x: e.clientX - dragStart.current.x, y: e.clientY - dragStart.current.y }); 
            }}
            onMouseUp={(e) => { 
              isDragging.current = false; 
              (e.target as HTMLElement).style.cursor = 'grab';
            }}
            onMouseLeave={(e) => { 
              isDragging.current = false; 
              (e.target as HTMLElement).style.cursor = 'grab';
            }}
          >
            <div style={{ position: "absolute", top: "16px", left: "20px", fontSize: "1.1rem", fontWeight: 600, color: "#e2e8f0", zIndex: 10, pointerEvents: "none" }}>2D Sparse Map</div>
            
            {numFrames > 0 ? (
              <img 
                src={`${API_BASE}/api/2dmap/${timestep}`} 
                alt="2D Sparse Map" 
                onLoad={() => { isImageLoading.current = false; }}
                onError={() => { isImageLoading.current = false; }}
                style={{ 
                  width: "100%", 
                  height: "100%", 
                  objectFit: "contain",
                  transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                  transformOrigin: "center",
                  pointerEvents: "none" // Prevent img from capturing mouse events and breaking drag
                }}
              />
            ) : (
              <div style={{ color: "#64748b", textAlign: "center", marginTop: "40vh" }}>
                <Compass size={48} style={{ margin: "0 auto 12px auto", opacity: 0.5 }} />
                Please upload a video to plot the 2D Sparse Map
              </div>
            )}
          </div>

          {/* Floating Video Feed Overlay (Bottom Right) */}
          {numFrames > 0 && (
            <div style={{ position: "absolute", bottom: "24px", right: "24px", width: "320px", backgroundColor: "#1e2128", borderRadius: "12px", border: "1px solid #2d313a", padding: "8px", boxShadow: "0 10px 25px rgba(0,0,0,0.5)", zIndex: 10 }}>
              <div style={{ fontSize: "0.85rem", fontWeight: 600, color: "#e2e8f0", marginBottom: "8px" }}>Live Camera Feed</div>
              <div style={{ width: "100%", backgroundColor: "#000", borderRadius: "6px", overflow: "hidden", aspectRatio: "16/9" }}>
                <img 
                  src={`${API_BASE}/api/frame/${timestep}`} 
                  alt="Live Camera Frame" 
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                />
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
