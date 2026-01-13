import { useEffect, useRef, useState } from "react";
import { useRoute } from "wouter";
import { useAnalysis } from "@/hooks/use-analysis";
import { motion, AnimatePresence } from "framer-motion";
import { TerminalProgressBar } from "@/components/TerminalProgressBar";
import { TerminalButton } from "@/components/TerminalButton";
import { MatrixRain } from "@/components/MatrixRain";
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
  const logsContainerRef = useRef<HTMLDivElement>(null);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
  
  // Handle manual scroll to disable/enable auto-scroll
  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
    // If user is within 50px of bottom, enable auto-scroll
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
    setShouldAutoScroll(isAtBottom);
  };

  // Auto-scroll logs only if shouldAutoScroll is true
  useEffect(() => {
    if (shouldAutoScroll) {
      logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [data?.analysis.logs, shouldAutoScroll]);

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
    <div className="min-h-screen bg-black text-[#00FF41] font-mono p-8 overflow-hidden relative flex flex-col">
      <div className="scanlines" />
      <div className="crt-flicker" />
      <MatrixRain />

      {/* Global Application Frame */}
      <div className="flex-1 border-2 border-[#00FF41] bg-black/40 backdrop-blur-sm relative flex flex-col p-1">
        {/* Top Header Frame Bar */}
        <div className="border border-[#00FF41] p-4 mb-4 flex justify-between items-center bg-[#001100]">
          <div className="flex items-center gap-6">
            <h1 className="text-3xl font-black tracking-[0.3em] text-glow">
              DNA.GENERATOR_V4
            </h1>
            <div className="hidden md:block text-[10px] text-[#008F11] tracking-widest uppercase">
              MILITARY GRADE ART ANALYSIS PROTOCOL
            </div>
          </div>
          <div className="flex gap-8 text-[10px] font-bold">
            <div className="flex flex-col items-end">
              <span className="text-[#008F11]">MET_API:</span>
              <span className="text-[#00FF41]">ONLINE</span>
            </div>
            <div className="flex flex-col items-end">
              <span className="text-[#008F11]">AIC_API:</span>
              <span className="text-[#00FF41]">ONLINE</span>
            </div>
            <div className="flex flex-col items-end">
              <span className="text-[#008F11]">CMA_API:</span>
              <span className="text-[#00FF41]">ONLINE</span>
            </div>
            <div className="flex flex-col items-end">
              <span className="text-[#008F11]">SYS_STATUS:</span>
              <span className="animate-pulse">ONLINE</span>
            </div>
            <div className="flex flex-col items-end">
              <span className="text-[#008F11]">SECURE_CONN:</span>
              <span>ENCRYPTED</span>
            </div>
            <div className="flex flex-col items-end">
              <span className="text-[#008F11]">VER:</span>
              <span>4.0.2</span>
            </div>
          </div>
        </div>

        <div className="flex-1 grid md:grid-cols-2 gap-6 p-4 overflow-hidden">
          {/* LEFT PANEL: DATA STREAM (Artworks) */}
          <div className="border border-[#00FF41] bg-black/60 p-1 flex flex-col overflow-hidden">
            <div className="bg-[#001100] border border-[#003300] p-6 flex-1 flex flex-col overflow-hidden">
              <div className="flex items-center gap-3 mb-6 border-b border-[#003300] pb-4">
                <div className="w-2 h-2 rounded-full bg-[#00FF41] animate-pulse" />
                <h2 className="text-sm font-bold tracking-[0.2em] text-glow uppercase">
                  INPUT_STREAM ({artworks.length}/30)
                </h2>
              </div>
              
              <div className="flex-1 overflow-y-auto space-y-3 custom-scrollbar pr-2">
                <AnimatePresence>
                  {artworks.map((art, i) => (
                    <motion.div
                      key={art.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.05 }}
                      className={`flex gap-3 p-3 border border-[#003300] ${art.isAnalyzed ? 'bg-[#001100]' : 'bg-black'} hover:border-[#00FF41] transition-colors group`}
                    >
                      <div className="w-16 h-16 bg-[#002200] border border-[#003300] relative overflow-hidden flex-shrink-0">
                        {art.imageUrl ? (
                          <img 
                            src={art.imageUrl} 
                            alt={art.title}
                            className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity grayscale hover:grayscale-0"
                          />
                        ) : (
                          <ImageIcon className="w-8 h-8 text-[#003300] absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                        )}
                        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-transparent via-[#00FF41]/20 to-transparent h-1 animate-scanline" />
                      </div>
                      <div className="flex-1 min-w-0 flex flex-col justify-center">
                        <div className="text-sm font-bold truncate text-[#00CC33]">{art.title}</div>
                        <div className="text-[11px] text-[#006600] truncate flex justify-between mt-1">
                          <span>{art.year || "UNKNOWN"}</span>
                          <span>{art.museumSource}</span>
                        </div>
                        <div className="flex items-center gap-2 mt-2">
                          <div className={`w-2 h-2 rounded-full ${art.isAnalyzed ? 'bg-[#00FF41] shadow-[0_0_5px_#00FF41]' : 'bg-[#003300]'}`} />
                          <span className="text-[10px] text-[#008F11] tracking-tighter">
                            {art.isAnalyzed ? "ANALYZED" : "QUEUED"}
                          </span>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                  
                  {artworks.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center text-[#003300] opacity-50 py-12">
                      <Loader2 className="w-12 h-12 animate-spin mb-4" />
                      <span className="text-sm tracking-widest">AWAITING DATA STREAM...</span>
                    </div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </div>

          {/* RIGHT PANEL: SYSTEM METRICS & LOGS */}
          <div className="border border-[#00FF41] bg-black/60 p-1 flex flex-col overflow-hidden">
            <div className="bg-[#001100] border border-[#003300] p-6 flex-1 flex flex-col overflow-hidden">
              <div className="flex items-center gap-3 mb-6 border-b border-[#003300] pb-4">
                <div className="w-2 h-2 rounded-full bg-[#00FF41] animate-pulse" />
                <h2 className="text-sm font-bold tracking-[0.2em] text-glow uppercase">
                  SYSTEM_METRICS
                </h2>
              </div>

              <div className="grid grid-cols-2 gap-6 mb-8">
                {[
                  { label: "DNA ARTISTIQUE", key: "artistic" },
                  { label: "COMPOSITION", key: "composition" },
                  { label: "LUMIÈRE", key: "light" },
                  { label: "COULEURS", key: "color" },
                  { label: "FINITIONS", key: "finish" },
                  { label: "ICONOGRAPHIE", key: "iconography" }
                ].map((item, i) => {
                  const frameworkResult = frameworkResults.find(r => r.frameworkType === item.key);
                  const status = frameworkResult?.status || (analysis.status === "completed" ? "completed" : "pending");
                  const progress = getFrameworkProgress(item.key);
                  
                  return (
                    <div key={item.key} className="flex flex-col gap-1">
                      <TerminalProgressBar 
                        label={`${i + 1}. ${item.label}`} 
                        progress={progress} 
                      />
                      <div className="flex justify-between text-[9px] px-1 uppercase tracking-tighter">
                        <span className={status === "completed" ? "text-[#00FF41]" : status === "failed" ? "text-red-500" : "text-[#008F11]"}>
                          STATUS: {status}
                        </span>
                        {status === "completed" && <span className="text-[#00FF41]">EXTRACTION COMPLETE</span>}
                        {status === "failed" && <span className="text-red-500">CORRUPTION DETECTED</span>}
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="flex-1 flex flex-col border-t border-[#003300] pt-4 overflow-hidden">
                <div className="flex justify-between text-[10px] text-[#008F11] mb-2 uppercase font-bold tracking-widest">
                  <span>SYSTEM LOG // LIVE STREAM</span>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    <span>REC</span>
                  </div>
                </div>
                <div 
                  ref={logsContainerRef}
                  onScroll={handleScroll}
                  className="bg-black/50 border border-[#003300] p-4 flex-1 text-[11px] text-[#00FF41] overflow-y-auto font-mono custom-scrollbar"
                >
                  {(analysis.logs || []).map((log, i) => (
                    <LogEntry key={i} text={log} />
                  ))}
                  <div ref={logsEndRef} />
                  {!isComplete && !isFailed && (
                    <div className="flex gap-2 mt-2">
                      <span className="text-[#008F11] animate-pulse">{">"}</span>
                      <span className="w-2 h-4 bg-[#00FF41] animate-pulse" />
                    </div>
                  )}
                </div>
              </div>

              {/* Action Area */}
              <div className="mt-6 flex items-center justify-between border-t border-[#003300] pt-4">
                <div className="text-[11px] text-[#008F11] font-bold tracking-widest">
                  {isComplete ? "SEQUENCE COMPLETE. PACKAGE READY." : isFailed ? "MISSION FAILED. ABORT." : "EXTRACTING DNA..."}
                </div>
                
                <AnimatePresence>
                  {isComplete && (
                    <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}>
                      <TerminalButton onClick={handleDownload} className="flex items-center gap-2">
                        <Download className="w-4 h-4" />
                        DOWNLOAD
                      </TerminalButton>
                    </motion.div>
                  )}
                  {isFailed && (
                    <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}>
                       <TerminalButton variant="danger" onClick={() => window.location.href = "/"}>
                        RESTART
                      </TerminalButton>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </div>
        </div>

        {/* Footer Frame Bar */}
        <div className="border-t border-[#00FF41] p-2 text-center text-[#003300] text-[8px] tracking-[0.5em] uppercase">
          RESTRICTED ACCESS // AUTHORIZED PERSONNEL ONLY // SESSION ID: IHL18
        </div>
      </div>
    </div>
  );
}
