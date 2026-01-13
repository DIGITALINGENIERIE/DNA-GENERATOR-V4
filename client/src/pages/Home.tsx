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
    <div className="min-h-screen bg-black text-[#00FF41] font-mono relative overflow-hidden flex flex-col items-center justify-center p-4">
      <div className="scanlines" />
      <div className="crt-flicker" />
      <MatrixRain />

      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-4xl z-10 grid md:grid-cols-2 gap-6"
      >
        {/* Left Side: Input Protocol */}
        <div className="border border-[#00FF41] bg-black/80 backdrop-blur-md p-1 ascii-border">
          <div className="bg-[#001100] border border-[#003300] p-6 h-full">
            <div className="flex items-center gap-3 mb-6 border-b border-[#003300] pb-4">
              <div className="w-2 h-2 rounded-full bg-[#00FF41] animate-pulse" />
              <h2 className="text-sm font-bold tracking-[0.2em] text-glow uppercase">
                INPUT_PROTOCOL
              </h2>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              <TerminalInput
                label="TARGET SUBJECT / ARTIST NAME"
                placeholder="ENTER ARTIST IDENTITY..."
                value={artistName}
                onChange={(e) => setArtistName(e.target.value)}
                disabled={isPending}
                autoFocus
              />

              <div className="space-y-2">
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
                <div className="relative">
                  <textarea
                    className="w-full h-64 bg-black/50 border border-[#003300] p-3 text-sm text-[#00FF41] focus:outline-none focus:border-[#00FF41] transition-colors resize-none font-mono placeholder:text-[#003300]"
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

        {/* Right Side: System Metrics (Placeholders for now) */}
        <div className="border border-[#00FF41] bg-black/80 backdrop-blur-md p-1 ascii-border">
          <div className="bg-[#001100] border border-[#003300] p-6 h-full flex flex-col">
            <div className="flex items-center gap-3 mb-6 border-b border-[#003300] pb-4">
              <div className="w-2 h-2 rounded-full bg-[#00FF41] animate-pulse" />
              <h2 className="text-sm font-bold tracking-[0.2em] text-glow uppercase">
                SYSTEM_METRICS
              </h2>
            </div>

            <div className="grid grid-cols-1 gap-6 flex-1">
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

            <div className="mt-6 border-t border-[#003300] pt-4">
              <div className="flex justify-between text-[8px] text-[#003300] mb-2 uppercase">
                <span>SYSTEM LOG</span>
                <span>LIVE STREAM</span>
              </div>
              <div className="bg-black/50 border border-[#003300] p-3 h-32 text-[10px] text-[#003300] overflow-hidden font-mono">
                Waiting for mission initialization...
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      <div className="mt-8 text-center text-[#003300] text-[8px] tracking-[0.5em] uppercase">
        RESTRICTED ACCESS // AUTHORIZED PERSONNEL ONLY // SESSION ID: IHL18
      </div>
    </div>
  );
}
