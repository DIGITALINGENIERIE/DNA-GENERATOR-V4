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
  const totalFrameworks = FRAMEWORK_TYPES.length;
  let frameworksCompleted = 0;

  for (const framework of FRAMEWORK_TYPES) {
    // Update log
    await storage.updateAnalysisStatus(
      analysisId, 
      "analyzing", 
      Math.floor((frameworksCompleted / totalFrameworks) * 100),
      `Running ${framework.toUpperCase()} PROTOCOL...`
    );

    // Mock individual analysis for speed (real vision API on 30 images * 6 frameworks is too slow/expensive for this demo)
    // In a real production app, we would queue these jobs.
    // Here we will generate the SYNTHESIS directly using the artwork titles/years as context for the LLM.
    
    const context = artworks.map(a => `- ${a.title} (${a.year})`).join("\n");
    
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

    } catch (error) {
      console.error(`Error in framework ${framework}:`, error);
      await storage.createFrameworkResult({
        analysisId,
        frameworkType: framework,
        synthesisText: "DATA CORRUPTION DETECTED. ANALYSIS FAILED.",
        status: "failed"
      });
    }

    frameworksCompleted++;
  }
}
