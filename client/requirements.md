## Packages
framer-motion | Complex animations for terminal text typing and scanlines
clsx | Utility for constructing className strings conditionally
tailwind-merge | Utility for merging Tailwind classes safely

## Notes
Tailwind Config - extend fontFamily:
fontFamily: {
  mono: ["'Fira Code'", "monospace"],
  terminal: ["'VT323'", "monospace"], // Fallback to a simpler mono if needed, but Fira Code is primary
}
colors: {
  terminal: {
    black: "#000000",
    green: "#00FF41",
    dim: "#003B00",
    alert: "#FF0000",
    warning: "#FFA500",
  }
}
