import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { api } from "@shared/routes";
import { z } from "zod";
import { performAnalysis } from "./services/analysis";
import JSZip from "jszip";
import { registerChatRoutes } from "./replit_integrations/chat";
import { registerImageRoutes } from "./replit_integrations/image";

// Simplified mock search for artwork images
async function fetchArtworksByTitle(artistName: string, title: string) {
  return [{
    title,
    year: "Unknown",
    imageUrl: "https://images.metmuseum.org/CRDImages/ep/original/DT1567.jpg",
    source: "Met",
    metadata: {}
  }];
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  // Register OpenAI integration routes
  registerChatRoutes(app);
  registerImageRoutes(app);

  // === ANALYSES ===

  app.post(api.analyses.create.path, async (req, res) => {
    try {
      const input = api.analyses.create.input.parse(req.body);
      const analysis = await storage.createAnalysis({ 
        artistName: input.artistName,
        artworkTitles: input.artworkTitles
      });
      
      const artworkTitles = input.artworkTitles
        .split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0);

      processAnalysis(analysis.id, input.artistName, artworkTitles).catch(err => {
        console.error("Analysis process failed:", err);
        storage.updateAnalysisStatus(analysis.id, "failed", undefined, `System critical failure: ${err.message}`);
      });

      res.status(201).json(analysis);
    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({
          message: err.errors[0].message,
          field: err.errors[0].path.join('.'),
        });
      }
      throw err;
    }
  });

  app.get(api.analyses.get.path, async (req, res) => {
    const id = Number(req.params.id);
    const analysis = await storage.getAnalysis(id);
    if (!analysis) {
      return res.status(404).json({ message: "Analysis not found" });
    }
    const artworks = await storage.getArtworksByAnalysisId(id);
    const frameworkResults = await storage.getFrameworkResultsByAnalysisId(id);
    
    res.json({ analysis, artworks, frameworkResults });
  });

  app.get(api.analyses.download.path, async (req, res) => {
    const id = Number(req.params.id);
    const analysis = await storage.getAnalysis(id);
    if (!analysis) {
      return res.status(404).json({ message: "Analysis not found" });
    }

    const results = await storage.getFrameworkResultsByAnalysisId(id);
    const artworks = await storage.getArtworksByAnalysisId(id);
    const zip = new JSZip();
    const folderName = `${analysis.artistName.replace(/\s+/g, '_')}_DNA_ANALYSIS_${new Date().toISOString().split('T')[0]}`;
    const folder = zip.folder(folderName);

    // Add result files
    results.forEach(result => {
      const filename = getFilenameForFramework(result.frameworkType);
      if (folder && result.synthesisText) {
        folder.file(filename, result.synthesisText);
      }
    });

    // Add artworks folder
    const artworksFolder = folder?.folder("ARTWORKS_CORPUS");
    if (artworksFolder) {
      const axios = (await import("axios")).default;
      const downloadPromises = artworks.map(async (art, i) => {
        try {
          const response = await axios.get(art.imageUrl, { 
            responseType: 'arraybuffer',
            timeout: 10000 
          });
          const extension = art.imageUrl.split('.').pop()?.split('?')[0] || 'jpg';
          const filename = `${(i + 1).toString().padStart(2, '0')}_${art.title.replace(/[^a-z0-9]/gi, '_').substring(0, 50)}.${extension}`;
          artworksFolder.file(filename, response.data);
        } catch (error) {
          console.error(`Failed to download image for artwork ${art.id}:`, error);
        }
      });
      await Promise.all(downloadPromises);
    }

    if (folder) {
      folder.file("README.txt", generateReadme(analysis.artistName, results.length));
    }

    const content = await zip.generateAsync({ type: "nodebuffer" });
    
    res.setHeader("Content-Type", "application/zip");
    res.setHeader("Content-Disposition", `attachment; filename=${folderName}.zip`);
    res.send(content);
  });

  return httpServer;
}

async function processAnalysis(analysisId: number, artistName: string, providedTitles: string[]) {
  try {
    await storage.updateAnalysisStatus(analysisId, "fetching_artworks", 0, "Initializing artwork retrieval protocols for provided corpus...");
    
    const results: any[] = [];
    for (const title of providedTitles) {
      const searchResults = await fetchArtworksByTitle(artistName, title);
      if (searchResults.length > 0) {
        results.push(searchResults[0]);
      }
    }
    
    if (results.length === 0) {
      await storage.updateAnalysisStatus(analysisId, "failed", 0, "No valid artwork images found for the provided titles.");
      return;
    }

    await storage.updateAnalysisStatus(analysisId, "fetching_artworks", 100, `Retrieved ${results.length} valid images. Validating assets...`);

    for (const art of results) {
      await storage.createArtwork({
        analysisId,
        title: art.title,
        year: art.year,
        imageUrl: art.imageUrl,
        museumSource: art.source,
        metadata: art.metadata,
      });
    }

    await storage.updateAnalysisStatus(analysisId, "analyzing", 0, "Initiating DNA extraction sequence...");
    await performAnalysis(analysisId, artistName);

    await storage.updateAnalysisStatus(analysisId, "completed", 100, "Sequence complete. Package ready for extraction.");

  } catch (error: any) {
    console.error("Background process error:", error);
    await storage.updateAnalysisStatus(analysisId, "failed", undefined, `Critical error: ${error.message}`);
  }
}

function getFilenameForFramework(type: string): string {
  const map: Record<string, string> = {
    artistic: "01_ADN_ARTISTIQUE.txt",
    composition: "02_ADN_COMPOSITION.txt",
    light: "03_ADN_LUMIERE.txt",
    color: "04_ADN_COULEURS.txt",
    finish: "05_ADN_FINITIONS.txt",
    iconography: "06_ADN_ICONOGRAPHIE.txt",
  };
  return map[type] || `${type}.txt`;
}

function generateReadme(artist: string, count: number): string {
  return `DNA.GENERATOR_V4 - ANALYSIS PACKAGE
═══════════════════════════════════════════

ARTIST: ${artist}
ANALYSIS DATE: ${new Date().toISOString()}
CORPUS: 30 Artworks (Target)
VERSION: 4.0.2

PROTOCOL: MILITARY GRADE ART ANALYSIS
CLASSIFICATION: RESTRICTED`;
}
