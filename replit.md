# DNA.GENERATOR_V4

## Overview
An art analysis application that scans artworks across major museum databases (Met, Art Institute, Cleveland) to extract stylistic fingerprints using AI. The app has a retro terminal/CRT aesthetic with Matrix-style animations.

## Tech Stack
- **Frontend**: React 18 with TypeScript, Vite, TailwindCSS
- **Backend**: Express.js with TypeScript
- **Database**: PostgreSQL with Drizzle ORM
- **AI**: OpenAI API via Replit AI Integrations

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
- `DATABASE_URL` - PostgreSQL connection string (auto-configured)
- `AI_INTEGRATIONS_OPENAI_API_KEY` - OpenAI API key (auto-configured via Replit AI Integrations)
- `AI_INTEGRATIONS_OPENAI_BASE_URL` - OpenAI base URL (auto-configured)

## Key Features
- Artist DNA extraction and analysis
- 6 analysis frameworks: artistic, composition, light, color, finish, iconography
- Real-time progress tracking
- Retro CRT terminal aesthetic
