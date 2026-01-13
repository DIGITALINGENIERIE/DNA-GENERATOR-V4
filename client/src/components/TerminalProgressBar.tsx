import { motion } from "framer-motion";

interface TerminalProgressBarProps {
  label: string;
  progress: number; // 0-100
  color?: string;
  showValue?: boolean;
}

export function TerminalProgressBar({
  label,
  progress,
  color = "#00FF41",
  showValue = true,
}: TerminalProgressBarProps) {
  // Create blocks for the progress bar
  const totalBlocks = 20;
  const filledBlocks = Math.round((progress / 100) * totalBlocks);

  return (
    <div className="mb-4">
      <div className="flex justify-between items-end mb-1 text-xs font-bold tracking-wider">
        <span className="text-[#00FF41] uppercase">{label}</span>
        {showValue && <span className="text-[#00FF41]">{Math.round(progress)}%</span>}
      </div>
      <div className="flex gap-[2px] h-4">
        {Array.from({ length: totalBlocks }).map((_, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0 }}
            animate={{ 
              opacity: i < filledBlocks ? 1 : 0.2,
              backgroundColor: i < filledBlocks ? color : "#003300"
            }}
            transition={{ duration: 0.1, delay: i * 0.02 }}
            className="flex-1 h-full skew-x-[-10deg]"
          />
        ))}
      </div>
    </div>
  );
}
