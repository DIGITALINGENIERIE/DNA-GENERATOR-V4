import React from "react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface TerminalButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: "primary" | "secondary" | "danger";
  isBlinking?: boolean;
}

export function TerminalButton({
  children,
  className,
  variant = "primary",
  isBlinking = false,
  ...props
}: TerminalButtonProps) {
  const variants = {
    primary: "border-[#00FF41] text-[#00FF41] hover:bg-[#00FF41] hover:text-black",
    secondary: "border-[#008F11] text-[#008F11] hover:bg-[#008F11] hover:text-black",
    danger: "border-red-500 text-red-500 hover:bg-red-500 hover:text-black",
  };

  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={cn(
        "relative px-6 py-3 font-mono font-bold uppercase tracking-widest text-sm transition-all duration-100",
        "border-2 bg-transparent focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-black",
        "disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-[#00FF41]",
        variants[variant],
        isBlinking && "animate-pulse",
        className
      )}
      {...props}
    >
      <span className="relative z-10 flex items-center justify-center gap-2">
        {isBlinking && <span className="w-2 h-2 bg-current block animate-ping" />}
        {children}
      </span>
      
      {/* Decorative corners */}
      <div className="absolute top-0 left-0 w-1 h-1 bg-current" />
      <div className="absolute top-0 right-0 w-1 h-1 bg-current" />
      <div className="absolute bottom-0 left-0 w-1 h-1 bg-current" />
      <div className="absolute bottom-0 right-0 w-1 h-1 bg-current" />
    </motion.button>
  );
}
