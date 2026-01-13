import { db } from "./db";
import {
  analyses, artworks, frameworkResults,
  type Analysis, type InsertAnalysis,
  type Artwork, type InsertArtwork,
  type FrameworkResult,
  type FrameworkType,
  type AnalysisStatus
} from "@shared/schema";
import { eq, desc } from "drizzle-orm";

export interface IStorage {
  // Analyses
  createAnalysis(analysis: InsertAnalysis): Promise<Analysis>;
  getAnalysis(id: number): Promise<Analysis | undefined>;
  updateAnalysisStatus(id: number, status: AnalysisStatus, progress?: number, log?: string): Promise<Analysis>;
  getAnalyses(): Promise<Analysis[]>;

  // Artworks
  createArtwork(artwork: InsertArtwork): Promise<Artwork>;
  getArtworksByAnalysisId(analysisId: number): Promise<Artwork[]>;
  updateArtworkAnalyzed(id: number, isAnalyzed: boolean): Promise<Artwork>;

  // Framework Results
  createFrameworkResult(result: { analysisId: number; frameworkType: FrameworkType; individualAnalyses?: any; synthesisText?: string; status?: string }): Promise<FrameworkResult>;
  getFrameworkResultsByAnalysisId(analysisId: number): Promise<FrameworkResult[]>;
  updateFrameworkResult(id: number, updates: Partial<FrameworkResult>): Promise<FrameworkResult>;
  getFrameworkResult(analysisId: number, type: FrameworkType): Promise<FrameworkResult | undefined>;
}

export class DatabaseStorage implements IStorage {
  async createAnalysis(insertAnalysis: InsertAnalysis): Promise<Analysis> {
    const [analysis] = await db.insert(analyses).values(insertAnalysis).returning();
    return analysis;
  }

  async getAnalysis(id: number): Promise<Analysis | undefined> {
    const [analysis] = await db.select().from(analyses).where(eq(analyses.id, id));
    return analysis;
  }

  async updateAnalysisStatus(id: number, status: AnalysisStatus, progress?: number, log?: string): Promise<Analysis> {
    const updates: any = { status };
    if (progress !== undefined) updates.progress = progress;
    
    // We need to append to logs, but for simplicity in this turn we might just fetch and update or use sql append if supported easily.
    // Let's just fetch-update for simplicity with drizzle.
    const current = await this.getAnalysis(id);
    const currentLogs = current?.logs || [];
    if (log) currentLogs.push(log);

    const [updated] = await db.update(analyses)
      .set({ ...updates, logs: currentLogs })
      .where(eq(analyses.id, id))
      .returning();
    return updated;
  }

  async getAnalyses(): Promise<Analysis[]> {
    return db.select().from(analyses).orderBy(desc(analyses.createdAt));
  }

  async createArtwork(insertArtwork: InsertArtwork): Promise<Artwork> {
    const [artwork] = await db.insert(artworks).values(insertArtwork).returning();
    return artwork;
  }

  async getArtworksByAnalysisId(analysisId: number): Promise<Artwork[]> {
    return db.select().from(artworks).where(eq(artworks.analysisId, analysisId));
  }

  async updateArtworkAnalyzed(id: number, isAnalyzed: boolean): Promise<Artwork> {
    const [artwork] = await db.update(artworks)
      .set({ isAnalyzed })
      .where(eq(artworks.id, id))
      .returning();
    return artwork;
  }

  async createFrameworkResult(result: any): Promise<FrameworkResult> {
    const [res] = await db.insert(frameworkResults).values(result).returning();
    return res;
  }

  async getFrameworkResultsByAnalysisId(analysisId: number): Promise<FrameworkResult[]> {
    return db.select().from(frameworkResults).where(eq(frameworkResults.analysisId, analysisId));
  }

  async updateFrameworkResult(id: number, updates: Partial<FrameworkResult>): Promise<FrameworkResult> {
    const [res] = await db.update(frameworkResults)
      .set(updates)
      .where(eq(frameworkResults.id, id))
      .returning();
    return res;
  }

  async getFrameworkResult(analysisId: number, type: FrameworkType): Promise<FrameworkResult | undefined> {
    const [res] = await db.select().from(frameworkResults)
      .where(eq(frameworkResults.analysisId, analysisId))
      .where(eq(frameworkResults.frameworkType, type))
      .limit(1);
    return res;
  }
}

export const storage = new DatabaseStorage();
