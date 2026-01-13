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
  const { mutate, isPending } = useCreateAnalysis();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!artistName.trim()) return;
    mutate({ artistName });
  };

  return (
    <div className="min-h-screen bg-black text-[#00FF41] font-mono relative overflow-hidden flex flex-col items-center justify-center p-4">
      <div className="scanlines" />
      <div className="crt-flicker" />
      <MatrixRain />

      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-2xl z-10"
      >
        {/* Header Block */}
        <div className="border border-[#00FF41] bg-black/80 backdrop-blur-md p-1 mb-8 ascii-border">
          <div className="bg-[#001100] border border-[#003300] p-6">
            <div className="flex items-center justify-between mb-8 border-b border-[#003300] pb-4">
              <div className="flex items-center gap-3">
                <Terminal className="w-6 h-6 animate-pulse" />
                <h1 className="text-2xl font-bold tracking-[0.2em] text-glow">
                  DNA.GENERATOR_V4
                </h1>
              </div>
              <div className="text-xs text-[#003300] flex flex-col items-end">
                <span>SYS.STATUS: ONLINE</span>
                <span>SECURE.CONN: TRUE</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-8 text-xs text-[#008F11]">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4" />
                <span>AI VISION CORE: READY</span>
              </div>
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4" />
                <span>MUSEUM API LINK: ACTIVE</span>
              </div>
              <div className="flex items-center gap-2">
                <Eye className="w-4 h-4" />
                <span>VISUAL SYNTHESIS: STANDBY</span>
              </div>
            </div>

            <div className="mb-8 font-mono text-sm leading-relaxed text-[#00CC33]">
              <p className="mb-4 typing-cursor">
                INITIALIZING ARTISTIC DNA EXTRACTION PROTOCOL...
              </p>
              <p className="opacity-70">
                SYSTEM WILL SCAN 30 ARTWORKS ACROSS MAJOR MUSEUM DATABASES TO EXTRACT
                STYLISTIC FINGERPRINTS.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-8">
              <TerminalInput
                label="TARGET IDENTITY"
                placeholder="ENTER ARTIST NAME..."
                value={artistName}
                onChange={(e) => setArtistName(e.target.value)}
                disabled={isPending}
                autoFocus
              />

              <div className="flex justify-end">
                <TerminalButton
                  type="submit"
                  disabled={isPending || !artistName.trim()}
                  isBlinking={isPending}
                >
                  {isPending ? "INITIALIZING..." : "INITIATE SEQUENCE"}
                </TerminalButton>
              </div>
            </form>
          </div>
        </div>

        <div className="text-center text-[#003300] text-xs tracking-widest">
          AUTHORIZED PERSONNEL ONLY // CLASSIFIED LEVEL 4
        </div>
      </motion.div>
    </div>
  );
}
