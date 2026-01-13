import { useEffect, useRef, useState } from "react";
import { useRoute } from "wouter";
import { useAnalysis } from "@/hooks/use-analysis";
import { motion, AnimatePresence } from "framer-motion";
import { TerminalProgressBar } from "@/components/TerminalProgressBar";
import { TerminalButton } from "@/components/TerminalButton";
import { api, buildUrl } from "@shared/routes";
import { 
  Download, 
  Activity, 
  AlertCircle, 
  CheckCircle2, 
  Loader2, 
  Image as ImageIcon 
} from "lucide-react";
import { type AnalysisStatus, type Artwork } from "@shared/schema";

// Helper to format logs with timestamps
const LogEntry = ({ text }: { text: string }) => (
  <div className="flex gap-2 font-mono text-xs mb-1">
    <span className="text-[#008F11]">[{new Date().toLocaleTimeString()}]</span>
    <span className="text-[#00FF41]">{text}</span>
  </div>
);

export default function Analysis() {
  const [, params] = useRoute("/analysis/:id");
  const id = params ? parseInt(params.id) : undefined;
  const { data, isLoading, error } = useAnalysis(id);
  const logsEndRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [data?.analysis.logs]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-[#00FF41] font-mono animate-pulse flex flex-col items-center gap-4">
          <Loader2 className="w-12 h-12 animate-spin" />
          <div className="tracking-widest">ESTABLISHING SECURE LINK...</div>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center p-4">
        <div className="border border-red-500 p-8 max-w-md w-full text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h1 className="text-red-500 font-bold text-xl mb-2">SYSTEM ERROR</h1>
          <p className="text-red-400 font-mono text-sm mb-6">
            CONNECTION TERMINATED UNEXPECTEDLY
          </p>
          <TerminalButton variant="danger" onClick={() => window.location.href = "/"}>
            RETURN TO ROOT
          </TerminalButton>
        </div>
      </div>
    );
  }

  const { analysis, artworks, frameworkResults } = data;
  const isComplete = analysis.status === "completed";
  const isFailed = analysis.status === "failed";

  // Calculate detailed progress per framework
  // Just simulated based on overall status for visual flair if detailed data isn't ready
  const getFrameworkProgress = (type: string) => {
    if (isComplete) return 100;
    if (analysis.status === "pending" || analysis.status === "fetching_artworks") return 0;
    
    // If we have results for this framework, it's done
    const result = frameworkResults.find(r => r.frameworkType === type);
    if (result && result.status === "completed") return 100;
    
    // Otherwise it's simulated based on overall progress
    return Math.min(Math.max((analysis.progress || 0) + (Math.random() * 20 - 10), 0), 99);
  };

  const handleDownload = () => {
    if (!id) return;
    const url = buildUrl(api.analyses.download.path, { id });
    window.location.href = url;
  };

  return (
    <div className="min-h-screen bg-black text-[#00FF41] font-mono p-4 md:p-6 lg:p-8 overflow-hidden relative">
      <div className="scanlines" />
      <div className="crt-flicker" />

      {/* Top Bar */}
      <header className="flex justify-between items-center mb-6 border-b border-[#003300] pb-4">
        <div className="flex flex-col">
          <h1 className="text-xl font-bold tracking-widest text-glow">
            TARGET: {analysis.artistName}
          </h1>
          <div className="text-xs text-[#008F11] flex gap-4 mt-1">
            <span>ID: {analysis.id.toString().padStart(6, '0')}</span>
            <span>STATUS: <span className="animate-pulse">{analysis.status.toUpperCase()}</span></span>
          </div>
        </div>
        <div className="text-right hidden md:block">
          <div className="text-xs text-[#003300]">SYSTEM UPTIME</div>
          <div className="font-mono text-[#00FF41]">{new Date().toISOString().split('T')[0]}</div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-140px)]">
        
        {/* LEFT PANEL: DATA STREAM (Artworks) */}
        <section className="lg:col-span-4 flex flex-col gap-4 h-full">
          <div className="ascii-border h-full flex flex-col bg-[#000500] border border-[#003300] p-1">
            <div className="bg-[#001100] px-3 py-2 border-b border-[#003300] flex justify-between items-center">
              <span className="text-xs font-bold tracking-widest">INPUT_STREAM</span>
              <span className="text-xs text-[#008F11]">{artworks.length}/30</span>
            </div>
            
            <div className="flex-1 overflow-y-auto p-2 space-y-2 custom-scrollbar">
              <AnimatePresence>
                {artworks.map((art, i) => (
                  <motion.div
                    key={art.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className={`flex gap-3 p-2 border border-[#003300] ${art.isAnalyzed ? 'bg-[#001100]' : 'bg-black'} hover:border-[#00FF41] transition-colors group`}
                  >
                    <div className="w-12 h-12 bg-[#002200] border border-[#003300] relative overflow-hidden flex-shrink-0">
                      {art.imageUrl ? (
                        <img 
                          src={art.imageUrl} 
                          alt={art.title}
                          className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity grayscale hover:grayscale-0"
                        />
                      ) : (
                        <ImageIcon className="w-6 h-6 text-[#003300] absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                      )}
                      {/* Scanline over image */}
                      <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-transparent via-[#00FF41]/20 to-transparent h-1 animate-scanline" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-bold truncate text-[#00CC33]">{art.title}</div>
                      <div className="text-[10px] text-[#006600] truncate flex justify-between">
                        <span>{art.year || "UNKNOWN"}</span>
                        <span>{art.museumSource}</span>
                      </div>
                      <div className="flex items-center gap-1 mt-1">
                        <div className={`w-1.5 h-1.5 rounded-full ${art.isAnalyzed ? 'bg-[#00FF41] shadow-[0_0_5px_#00FF41]' : 'bg-[#003300]'}`} />
                        <span className="text-[10px] text-[#008F11] tracking-tighter">
                          {art.isAnalyzed ? "ANALYZED" : "QUEUED"}
                        </span>
                      </div>
                    </div>
                  </motion.div>
                ))}
                
                {artworks.length === 0 && (
                  <div className="h-full flex flex-col items-center justify-center text-[#003300] opacity-50">
                    <Loader2 className="w-8 h-8 animate-spin mb-2" />
                    <span className="text-xs tracking-widest">AWAITING DATA STREAM...</span>
                  </div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </section>

        {/* RIGHT PANEL: SYSTEM METRICS */}
        <section className="lg:col-span-8 flex flex-col gap-6 h-full">
          
          {/* Progress Section */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-[#000500] border border-[#003300] p-6 ascii-border">
            <div className="col-span-full mb-2 flex items-center gap-2 border-b border-[#003300] pb-2">
              <Activity className="w-4 h-4" />
              <span className="text-xs font-bold tracking-widest">ANALYSIS_METRICS</span>
            </div>

            <TerminalProgressBar 
              label="ARTISTIC DNA" 
              progress={getFrameworkProgress("artistic")} 
            />
            <TerminalProgressBar 
              label="COMPOSITION" 
              progress={getFrameworkProgress("composition")} 
            />
            <TerminalProgressBar 
              label="LIGHT & SHADOW" 
              progress={getFrameworkProgress("light")} 
            />
            <TerminalProgressBar 
              label="COLOR THEORY" 
              progress={getFrameworkProgress("color")} 
            />
            <TerminalProgressBar 
              label="SURFACE FINISH" 
              progress={getFrameworkProgress("finish")} 
            />
            <TerminalProgressBar 
              label="ICONOGRAPHY" 
              progress={getFrameworkProgress("iconography")} 
            />
          </div>

          {/* System Log */}
          <div className="flex-1 bg-black border border-[#003300] p-4 font-mono text-xs overflow-hidden flex flex-col relative ascii-border">
            <div className="absolute top-0 right-0 p-2 text-[10px] text-[#003300]">SYS.LOG</div>
            <div className="flex-1 overflow-y-auto space-y-1 custom-scrollbar pr-2">
              {(analysis.logs || []).map((log, i) => (
                <LogEntry key={i} text={log} />
              ))}
              <div ref={logsEndRef} />
              
              {!isComplete && !isFailed && (
                <div className="flex gap-2 mt-2 opacity-50">
                  <span className="text-[#008F11]">{">"}</span>
                  <span className="w-2 h-4 bg-[#00FF41] animate-pulse" />
                </div>
              )}
            </div>
          </div>

          {/* Action Area */}
          <div className="h-16 flex items-center justify-between">
            <div className="text-xs text-[#003300]">
              {isComplete ? "SEQUENCE COMPLETE. READY FOR EXPORT." : "PROCESSING SEQUENCE..."}
            </div>
            
            <AnimatePresence>
              {isComplete && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                >
                  <TerminalButton onClick={handleDownload} className="flex items-center gap-2">
                    <Download className="w-4 h-4" />
                    DOWNLOAD PACKAGE
                  </TerminalButton>
                </motion.div>
              )}
              
              {isFailed && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                >
                   <TerminalButton variant="danger" onClick={() => window.location.reload()}>
                    RETRY SEQUENCE
                  </TerminalButton>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

        </section>
      </div>
    </div>
  );
}
