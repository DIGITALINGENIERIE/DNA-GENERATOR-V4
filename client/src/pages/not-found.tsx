import { TerminalButton } from "@/components/TerminalButton";
import { AlertCircle } from "lucide-react";

export default function NotFound() {
  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-4">
      <div className="border border-[#00FF41] p-8 max-w-md w-full text-center relative scanlines">
        {/* Corner accents */}
        <div className="absolute top-0 left-0 w-2 h-2 border-t-2 border-l-2 border-[#00FF41]" />
        <div className="absolute top-0 right-0 w-2 h-2 border-t-2 border-r-2 border-[#00FF41]" />
        <div className="absolute bottom-0 left-0 w-2 h-2 border-b-2 border-l-2 border-[#00FF41]" />
        <div className="absolute bottom-0 right-0 w-2 h-2 border-b-2 border-r-2 border-[#00FF41]" />

        <AlertCircle className="w-16 h-16 text-[#00FF41] mx-auto mb-6 animate-pulse" />
        
        <h1 className="text-[#00FF41] font-mono font-bold text-4xl mb-2 tracking-widest text-glow">
          404
        </h1>
        
        <p className="text-[#008F11] font-mono text-sm mb-8 uppercase tracking-wider">
          Sector Not Found<br/>
          Navigation Path Invalid
        </p>
        
        <TerminalButton onClick={() => window.location.href = "/"}>
          RETURN TO BASE
        </TerminalButton>
      </div>
    </div>
  );
}
