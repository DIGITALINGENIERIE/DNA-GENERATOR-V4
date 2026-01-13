import React from "react";
import { cn } from "@/lib/utils";

interface TerminalInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
}

export const TerminalInput = React.forwardRef<HTMLInputElement, TerminalInputProps>(
  ({ className, label, ...props }, ref) => {
    return (
      <div className="w-full">
        {label && (
          <label className="block mb-2 text-xs font-bold tracking-widest uppercase text-[#008F11]">
            {">"} {label}
          </label>
        )}
        <div className="relative group">
          <input
            ref={ref}
            className={cn(
              "w-full bg-black/50 border-b-2 border-[#003300] py-2 px-0 text-[#00FF41] font-mono text-lg uppercase placeholder:text-[#003300]",
              "focus:outline-none focus:border-[#00FF41] focus:bg-[#001100]/30 transition-colors",
              className
            )}
            autoComplete="off"
            {...props}
          />
          <div className="absolute bottom-0 left-0 h-[2px] w-0 bg-[#00FF41] transition-all duration-300 group-focus-within:w-full" />
        </div>
      </div>
    );
  }
);
TerminalInput.displayName = "TerminalInput";
