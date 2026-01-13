import { pgTable, text, serial, integer, boolean, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";
import { relations } from "drizzle-orm";

export * from "./models/chat";

// === TABLES ===

export const analyses = pgTable("analyses", {
  id: serial("id").primaryKey(),
  artistName: text("artist_name").notNull(),
  status: text("status").notNull().default("pending"), // pending, fetching_artworks, analyzing, synthesizing, completed, failed
  progress: integer("progress").default(0), // 0-100
  logs: text("logs").array().default([]),
  createdAt: timestamp("created_at").defaultNow(),
});

export const artworks = pgTable("artworks", {
  id: serial("id").primaryKey(),
  analysisId: integer("analysis_id").notNull(),
  title: text("title").notNull(),
  year: text("year"),
  imageUrl: text("image_url").notNull(),
  museumSource: text("museum_source").notNull(), // Met, Art Institute, Cleveland
  metadata: jsonb("metadata"),
  isAnalyzed: boolean("is_analyzed").default(false),
});

export const frameworkResults = pgTable("framework_results", {
  id: serial("id").primaryKey(),
  analysisId: integer("analysis_id").notNull(),
  frameworkType: text("framework_type").notNull(), // artistic, composition, light, color, finish, iconography
  individualAnalyses: jsonb("individual_analyses"), // Array of per-artwork analysis
  synthesisText: text("synthesis_text"), // The final text report
  status: text("status").default("pending"),
});

// === RELATIONS ===

export const analysesRelations = relations(analyses, ({ many }) => ({
  artworks: many(artworks),
  frameworkResults: many(frameworkResults),
}));

export const artworksRelations = relations(artworks, ({ one }) => ({
  analysis: one(analyses, {
    fields: [artworks.analysisId],
    references: [analyses.id],
  }),
}));

export const frameworkResultsRelations = relations(frameworkResults, ({ one }) => ({
  analysis: one(analyses, {
    fields: [frameworkResults.analysisId],
    references: [analyses.id],
  }),
}));

// === SCHEMAS ===

export const insertAnalysisSchema = createInsertSchema(analyses).pick({
  artistName: true,
}).extend({
  artworkTitles: z.string(), // Textarea input: one per line
});

export type AnalysisInput = z.infer<typeof insertAnalysisSchema>;

export const insertArtworkSchema = createInsertSchema(artworks).omit({
  id: true,
  isAnalyzed: true,
});

export const insertFrameworkResultSchema = createInsertSchema(frameworkResults).omit({
  id: true,
  status: true,
});

// === TYPES ===

export type Analysis = typeof analyses.$inferSelect;
export type InsertAnalysis = z.infer<typeof insertAnalysisSchema>;
export type Artwork = typeof artworks.$inferSelect;
export type InsertArtwork = z.infer<typeof insertArtworkSchema>;
export type FrameworkResult = typeof frameworkResults.$inferSelect;

export type AnalysisStatus = "pending" | "fetching_artworks" | "analyzing" | "synthesizing" | "completed" | "failed";

export const FRAMEWORK_TYPES = [
  "artistic",
  "composition",
  "light",
  "color",
  "finish",
  "iconography"
] as const;

export type FrameworkType = typeof FRAMEWORK_TYPES[number];
