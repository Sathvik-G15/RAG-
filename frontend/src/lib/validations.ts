import { z } from 'zod';

export const analyzeRequestSchema = z.object({
  query: z.string().min(1, 'Query is required').max(5000, 'Query too long'),
  age: z.number().int().min(0).max(150).optional().nullable(),
  gender: z.enum(['male', 'female', 'other']).optional().nullable(),
  comorbidities: z.array(z.string()).optional(),
  medications: z.array(z.string()).optional(),
  allergies: z.array(z.string()).optional(),
  vitals: z
    .object({
      temperature: z.number().optional(),
      heartRate: z.number().optional(),
      bloodPressureSystolic: z.number().optional(),
      bloodPressureDiastolic: z.number().optional(),
      respiratoryRate: z.number().optional(),
      oxygenSaturation: z.number().optional(),
    })
    .optional()
    .nullable(),
});

export const loginSchema = z.object({
  username: z.string().min(1, 'Username is required'),
  password: z.string().min(1, 'Password is required'),
});

export const feedbackSchema = z.object({
  query: z.string(),
  helpful: z.boolean(),
  comment: z.string().optional(),
});

export const guidelineUploadSchema = z.object({
  source: z.string().min(1, 'Source is required'),
  title: z.string().optional(),
  year: z.number().int().min(1900).max(new Date().getFullYear()).optional(),
  trustScore: z.number().min(0).max(1).default(1.0),
});

export const reviewSchema = z.object({
  hallucinationFlag: z.boolean().optional(),
  notes: z.string().optional(),
  reviewer: z.string().min(1, 'Reviewer name is required'),
});

export type AnalyzeRequest = z.infer<typeof analyzeRequestSchema>;
export type LoginRequest = z.infer<typeof loginSchema>;
export type FeedbackRequest = z.infer<typeof feedbackSchema>;
export type GuidelineUploadRequest = z.infer<typeof guidelineUploadSchema>;
export type ReviewRequest = z.infer<typeof reviewSchema>;

export const analyzeResponseSchema = z.object({
  primary_diagnosis: z.string().nullable(),
  confidence: z.number().min(0).max(1),
  uncertainty: z.number().min(0).max(1),
  hallucination_score: z.number().min(0).max(1),
  risk_level: z.enum(['low', 'medium', 'high', 'critical']),
  decision: z.enum(['diagnose', 'escalate', 'abstain']),
  escalated_reason: z.string().nullable(),
  retrieval_k_used: z.number().int().nonnegative(),
  retrieval_steps: z.number().int().nonnegative(),
  confidence_curve: z.array(z.number().min(0).max(1)),
  differential: z.array(
    z.object({
      diagnosis: z.string(),
      probability: z.number().min(0).max(1),
      sources: z.array(z.string()),
    })
  ),
  evidence: z.array(
    z.object({
      source: z.string(),
      title: z.string().nullable(),
      rank: z.number().int().nonnegative(),
      text: z.string(),
    })
  ),
  reasoning: z.string(),
  disclaimer: z.string(),
});

export type AnalyzeResponse = z.infer<typeof analyzeResponseSchema>;

export const corpusStatsSchema = z.object({
  total_chunks: z.number().int().nonnegative(),
  total_documents: z.number().int().nonnegative(),
  specialty_distribution: z.record(z.string(), z.number().int().nonnegative()),
  source_distribution: z.record(z.string(), z.number().int().nonnegative()),
  last_updated: z.string().datetime().nullable(),
});

export type CorpusStats = z.infer<typeof corpusStatsSchema>;

export const metricsSchema = z.object({
  method: z.string(),
  n: z.number().int().nonnegative(),
  accuracy: z.number().min(0).max(1),
  avg_confidence: z.number().min(0).max(1),
  avg_hallucination: z.number().min(0).max(1),
  avg_retrieval_k: z.number().min(0),
  budget_exhausted_rate: z.number().min(0).max(1),
  avg_latency_ms: z.number().min(0),
});

export type Metrics = z.infer<typeof metricsSchema>;