import { z } from 'zod';
import { insertAnalysisSchema, analyses, artworks, frameworkResults, FRAMEWORK_TYPES } from './schema';

export const errorSchemas = {
  validation: z.object({
    message: z.string(),
    field: z.string().optional(),
  }),
  notFound: z.object({
    message: z.string(),
  }),
  internal: z.object({
    message: z.string(),
  }),
};

export const api = {
  analyses: {
    create: {
      method: 'POST' as const,
      path: '/api/analyses',
      input: insertAnalysisSchema,
      responses: {
        201: z.custom<typeof analyses.$inferSelect>(),
        400: errorSchemas.validation,
      },
    },
    get: {
      method: 'GET' as const,
      path: '/api/analyses/:id',
      responses: {
        200: z.object({
          analysis: z.custom<typeof analyses.$inferSelect>(),
          artworks: z.array(z.custom<typeof artworks.$inferSelect>()),
          frameworkResults: z.array(z.custom<typeof frameworkResults.$inferSelect>()),
        }),
        404: errorSchemas.notFound,
      },
    },
    download: {
      method: 'GET' as const,
      path: '/api/analyses/:id/download',
      responses: {
        200: z.any(), // File download
        404: errorSchemas.notFound,
      },
    },
  },
};

export function buildUrl(path: string, params?: Record<string, string | number>): string {
  let url = path;
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (url.includes(`:${key}`)) {
        url = url.replace(`:${key}`, String(value));
      }
    });
  }
  return url;
}

export type AnalysisInput = z.infer<typeof api.analyses.create.input>;
export type AnalysisResponse = z.infer<typeof api.analyses.get.responses[200]>;
