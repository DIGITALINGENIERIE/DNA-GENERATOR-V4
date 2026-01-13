# DNA.GENERATOR_V4

## Overview
An art analysis application that scans artworks across major museum databases (Met, Art Institute, Cleveland) to extract stylistic fingerprints using AI. The app has a retro terminal/CRT aesthetic with Matrix-style animations.

## Tech Stack
- **Frontend**: React 18 with TypeScript, Vite, TailwindCSS
- **Backend**: Express.js with TypeScript
- **Database**: PostgreSQL with Drizzle ORM
- **AI**: OpenAI API (Hybrid: Replit AI Integrations + External OpenAI API Key Support)
- **Utilities**: JSZip for data extraction, p-retry for API stability

## Project Structure
```
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/    # UI components (Radix-based + custom)
│   │   ├── hooks/         # React hooks
│   │   ├── lib/           # Utilities and query client
│   │   └── pages/         # Route pages
│   └── index.html
├── server/                 # Express backend
│   ├── replit_integrations/  # AI integration modules
│   │   ├── batch/         # Batch processing with rate limiting
│   │   ├── chat/          # Chat API routes
│   │   └── image/         # Image generation routes
│   ├── services/          # Business logic
│   ├── db.ts              # Database connection
│   ├── routes.ts          # API routes
│   └── index.ts           # Server entry point
├── shared/                 # Shared types and schemas
│   ├── models/            # Database models
│   └── schema.ts          # Drizzle schema
└── script/                # Build scripts
```

## Running the Application
- Development: `npm run dev` - Runs on port 5000
- Build: `npm run build` - Builds client and server
- Production: `npm run start` - Runs production build
- Database: `npm run db:push` - Push schema changes

## Environment Variables
- `DATABASE_URL` - PostgreSQL connection string
- `OPENAI_API_KEY` - (Optional) External OpenAI key for enhanced stability
- `AI_INTEGRATIONS_OPENAI_API_KEY` - Replit AI API key (auto-fallback)
- `AI_INTEGRATIONS_OPENAI_BASE_URL` - Replit AI base URL

## Key Features
- **Artist DNA Extraction**: 6 specialized frameworks (Artistic, Composition, Light, Color, Finish, Iconography).
- **Batch Processing**: Military-grade stability with parallel processing and smart retries.
- **Real-time Dashboard**: Granular [XX/30] progress tracking and live system logs.
- **Secure Extraction**: Automated ZIP package generation containing:
  - 6 synthesized DNA reports (.txt)
  - Full artwork corpus imagery
  - Analysis README manifest
- **CRT Terminal Aesthetic**: Immersive Matrix-style UI with scanlines and flicker effects.
