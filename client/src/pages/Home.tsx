import { useState } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { TerminalInput } from "@/components/TerminalInput";
import { TerminalButton } from "@/components/TerminalButton";
import { MatrixRain } from "@/components/MatrixRain";
import { useCreateAnalysis } from "@/hooks/use-analysis";
import { Terminal, Cpu, Database, Eye } from "lucide-react";

export default function Home() {
  const [artistName, setArtistName] = useState("");
  const [artworkTitles, setArtworkTitles] = useState("");
  const { mutate, isPending } = useCreateAnalysis();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!artistName.trim() || !artworkTitles.trim()) return;
    mutate({ artistName, artworkTitles });
  };

  const lines = artworkTitles.split('\n').filter(l => l.trim().length > 0);
  const count = lines.length;

  return (
    <div className="min-h-screen bg-black text-[#00FF41] font-mono relative overflow-hidden p-8 flex flex-col">
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

        <div className="flex-1 grid md:grid-cols-2 gap-6 p-4">
          {/* Left Side: Input Protocol */}
          <div className="border border-[#00FF41] bg-black/60 p-1 flex flex-col">
            <div className="bg-[#001100] border border-[#003300] p-6 flex-1 flex flex-col">
              <div className="flex items-center gap-3 mb-6 border-b border-[#003300] pb-4">
                <div className="w-2 h-2 rounded-full bg-[#00FF41] animate-pulse" />
                <h2 className="text-sm font-bold tracking-[0.2em] text-glow uppercase">
                  INPUT_PROTOCOL
                </h2>
              </div>

              <form onSubmit={handleSubmit} className="flex-1 flex flex-col space-y-6">
                <TerminalInput
                  label="TARGET SUBJECT / ARTIST NAME"
                  placeholder="ENTER ARTIST IDENTITY..."
                  value={artistName}
                  onChange={(e) => setArtistName(e.target.value)}
                  disabled={isPending}
                  autoFocus
                />

                <div className="flex-1 flex flex-col space-y-2">
                  <div className="flex justify-between items-end">
                    <label className="text-xs text-[#008F11] font-bold tracking-widest uppercase">
                      CORPUS DATA (30 ARTWORKS)
                    </label>
                    <div className="border border-[#003300] px-2 py-0.5 bg-black/50">
                      <span className="text-[10px] text-[#008F11]">COUNT: </span>
                      <span className={count === 30 ? "text-[#00FF41]" : "text-red-500"}>
                        {count.toString().padStart(2, '0')}/30
                      </span>
                    </div>
                  </div>
                  <div className="relative flex-1">
                    <textarea
                      className="w-full h-full bg-black/50 border border-[#003300] p-3 text-sm text-[#00FF41] focus:outline-none focus:border-[#00FF41] transition-colors resize-none font-mono placeholder:text-[#003300]"
                      placeholder="1. Starry Night (1889)&#10;2. Sunflowers (1888)&#10;..."
                      value={artworkTitles}
                      onChange={(e) => setArtworkTitles(e.target.value)}
                      disabled={isPending}
                    />
                    <div className="absolute bottom-2 right-2 text-[8px] text-[#003300] pointer-events-none">
                      * FORMAT: ONE ARTWORK TITLE PER LINE. YEAR OPTIONAL. STRICT ADHERENCE REQUIRED.
                    </div>
                  </div>
                </div>

                <div className="flex justify-center pt-4">
                  <TerminalButton
                    type="submit"
                    disabled={isPending || !artistName.trim() || !artworkTitles.trim()}
                    isBlinking={isPending}
                    className="w-full py-4 text-lg"
                  >
                    {isPending ? "INITIALIZING..." : "INITIATE SEQUENCE ▶"}
                  </TerminalButton>
                </div>
              </form>
            </div>
          </div>

          {/* Right Side: System Metrics & Logs */}
          <div className="border border-[#00FF41] bg-black/60 p-1 flex flex-col">
            <div className="bg-[#001100] border border-[#003300] p-6 flex-1 flex flex-col">
              <div className="flex items-center gap-3 mb-6 border-b border-[#003300] pb-4">
                <div className="w-2 h-2 rounded-full bg-[#00FF41] animate-pulse" />
                <h2 className="text-sm font-bold tracking-[0.2em] text-glow uppercase">
                  SYSTEM_METRICS
                </h2>
              </div>

              <div className="grid grid-cols-2 gap-6 mb-8">
                {[
                  "DNA ARTISTIQUE", "COMPOSITION",
                  "LUMIÈRE", "COULEURS",
                  "FINITIONS", "ICONOGRAPHIE"
                ].map((label, i) => (
                  <div key={label} className="space-y-2 opacity-50">
                    <div className="flex justify-between text-[10px] tracking-tighter">
                      <span>{i + 1}. {label}</span>
                      <span>[00/30]</span>
                    </div>
                    <div className="h-4 bg-[#001100] border border-[#003300] flex gap-1 p-0.5">
                      {Array.from({ length: 15 }).map((_, j) => (
                        <div key={j} className="flex-1 bg-[#003300]/20" />
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              <div className="flex-1 flex flex-col border-t border-[#003300] pt-4">
                <div className="flex justify-between text-[8px] text-[#003300] mb-2 uppercase">
                  <span>SYSTEM LOG</span>
                  <span>LIVE STREAM</span>
                </div>
                <div className="bg-black/50 border border-[#003300] p-4 flex-1 text-[11px] text-[#00FF41] overflow-y-auto font-mono scrollbar-thin">
                  <div className="text-[#008F11] mb-2">[INIT] CORE SYSTEM READY...</div>
                  <div className="text-[#008F11] mb-2">[MEM] 128GB NEURAL BUFFER ALLOCATED</div>
                  <div className="text-[#008F11] mb-2">[NET] TUNNEL ESTABLISHED TO MUSEUM CLOUD</div>
                  <div className="animate-pulse">Waiting for mission initialization...</div>
                </div>
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
