import { storage } from "../storage";
import { FRAMEWORK_TYPES } from "@shared/schema";
import OpenAI from "openai";
import { batchProcess } from "../replit_integrations/batch";

// We use the integration's environment variables
const openai = new OpenAI({
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
});

export async function performAnalysis(analysisId: number, artistName: string) {
  const artworks = await storage.getArtworksByAnalysisId(analysisId);
  const context = artworks.map(a => `- ${a.title} (${a.year})`).join("\n");
  
  // Using the robust batch processor for AI calls
  await batchProcess(
    [...FRAMEWORK_TYPES],
    async (framework: any) => {
      try {
        const completion = await openai.chat.completions.create({
          model: "gpt-5", // Optimized for latest model
          messages: [
            {
              role: "system",
              content: `You are an expert art historian and analyst for the DNA.GENERATOR_V4 system.
              Task: Generate a deep 'Artistic DNA' report for the framework: ${framework.toUpperCase()}.
              
              Output structure:
              DNA.GENERATOR_V4 - ${framework.toUpperCase()} DNA EXTRACTION
              ARTIST: ${artistName}
              ═══════════════════════════════════════════
              
              [Generate 5 detailed paragraphs specific to ${framework} analysis]`
            },
            {
              role: "user",
              content: `Analyze this corpus for ${artistName}:\n${context}`
            }
          ],
          max_completion_tokens: 1500,
        });

        const synthesisText = completion.choices[0]?.message?.content || "ANALYSIS_ERROR: NO_DATA_RETURNED";

        await storage.createFrameworkResult({
          analysisId,
          frameworkType: framework,
          synthesisText,
          status: "completed"
        });

        const currentResults = await storage.getFrameworkResultsByAnalysisId(analysisId);
        const progress = Math.floor((currentResults.length / FRAMEWORK_TYPES.length) * 100);
        
        await storage.updateAnalysisStatus(
          analysisId, 
          progress === 100 ? "completed" : "analyzing", 
          progress,
          `Protocol ${framework.toUpperCase()} successfully extracted.`
        );

      } catch (error) {
        console.error(`[CRITICAL] Framework ${framework} extraction failed:`, error);
        await storage.createFrameworkResult({
          analysisId,
          frameworkType: framework,
          synthesisText: `DNA_CORRUPTION_DETECTED: ${error instanceof Error ? error.message : "UNKNOWN_SYSTEM_FAILURE"}`,
          status: "failed"
        });
      }
    },
    { concurrency: 2, retries: 3 }
  );
}
