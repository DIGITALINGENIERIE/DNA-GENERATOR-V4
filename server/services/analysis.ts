import { storage } from "../storage";
import { FRAMEWORK_TYPES } from "@shared/schema";
import OpenAI from "openai";

// We use the integration's environment variables
const openai = new OpenAI({
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
});

export async function performAnalysis(analysisId: number, artistName: string) {
  const artworks = await storage.getArtworksByAnalysisId(analysisId);
  const context = artworks.map(a => `- ${a.title} (${a.year})`).join("\n");
  
  // We process frameworks in parallel to reach "maximum power"
  await Promise.all(FRAMEWORK_TYPES.map(async (framework) => {
    try {
      const completion = await openai.chat.completions.create({
        model: "gpt-5.1",
        messages: [
          {
            role: "system",
            content: `You are an expert art historian and analyst. You are part of a military-grade art analysis system called DNA.GENERATOR_V4.
            Your task is to generate a synthesized 'Artistic DNA' report for the framework: ${framework.toUpperCase()}.
            
            Follow this exact format for the output:
            DNA.GENERATOR_V4 - ${framework.toUpperCase()} DNA EXTRACTION
            ARTIST: ${artistName}
            ═══════════════════════════════════════════

            [Generate 5 detailed sections specific to ${framework} analysis based on the artworks provided]
            `
          },
          {
            role: "user",
            content: `Analyze the following corpus of artworks for ${artistName}:\n${context}`
          }
        ],
        max_completion_tokens: 1000,
      });

      const synthesisText = completion.choices[0].message.content || "Analysis failed.";

      await storage.createFrameworkResult({
        analysisId,
        frameworkType: framework,
        synthesisText,
        status: "completed"
      });

      // Update progress in storage (simplified)
      const currentResults = await storage.getFrameworkResultsByAnalysisId(analysisId);
      const progress = Math.floor((currentResults.length / FRAMEWORK_TYPES.length) * 100);
      await storage.updateAnalysisStatus(
        analysisId, 
        progress === 100 ? "completed" : "analyzing", 
        progress,
        `Protocol ${framework.toUpperCase()} successfully extracted.`
      );

    } catch (error) {
      console.error(`Error in framework ${framework}:`, error);
      await storage.createFrameworkResult({
        analysisId,
        frameworkType: framework,
        synthesisText: "DATA CORRUPTION DETECTED. ANALYSIS FAILED.",
        status: "failed"
      });
    }
  }));
}
