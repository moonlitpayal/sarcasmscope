/**
 * SARcasmScope — Hybrid Contextual Engine Adapter
 *
 * Supports two operational deployment modes:
 * Mode A: Standalone Client-Side
 *   - Runs Stage 2 linguistic incongruity + OOF-trained WordLex/CharSub fusion directly in-browser
 *   - Zero server dependency, latency < 1ms, zero marginal cloud cost
 *
 * Mode B: Hybrid API Inference
 *   - Calls `/api/predict-contextual` for external transformer inference (MuRIL / XLM-R)
 *   - Fuses transformer probability with client-side linguistic & statistical features
 *   - Graceful timeout fallback to Mode A on offline or timeout errors
 */

import {
  predictHybrid,
  HybridPredictionResult,
  SemanticFeatures,
  DEFAULT_FUSION_WEIGHTS,
  DEFAULT_ABSTAIN_THRESHOLD,
} from "./hybridFusion";

export interface ContextualPredictionOptions {
  mode?: "standalone" | "hybrid";
  apiUrl?: string;
  timeoutMs?: number;
  temperature?: number;
  abstainThreshold?: number;
}

export interface ApiTransformerResponse {
  p_sarcastic: number;
  model_name: string;
  latency_ms: number;
  status: "success" | "error";
}

/**
 * Predicts sarcasm using either standalone calibrated hybrid fusion or remote contextual API.
 */
export async function predictContextual(
  text: string,
  context = "",
  options: ContextualPredictionOptions = {}
): Promise<HybridPredictionResult> {
  const mode = options.mode ?? "standalone";
  const apiUrl = options.apiUrl ?? "/api/predict-contextual";
  const timeoutMs = options.timeoutMs ?? 1500;
  const abstainThreshold = options.abstainThreshold ?? DEFAULT_ABSTAIN_THRESHOLD;
  const weights =
    options.temperature !== undefined
      ? { ...DEFAULT_FUSION_WEIGHTS, temperature: options.temperature }
      : DEFAULT_FUSION_WEIGHTS;

  if (mode === "standalone" || typeof window === "undefined") {
    return predictHybrid(text, context, undefined, weights, abstainThreshold);
  }

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    const res = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, context }),
      signal: controller.signal,
    });

    clearTimeout(timer);

    if (res.ok) {
      const data = (await res.json()) as ApiTransformerResponse;
      const externalSemantic: Partial<SemanticFeatures> = {
        p_transformer: data.p_sarcastic,
        embedding_source: "transformer_muril",
      };
      return predictHybrid(text, context, externalSemantic, weights, abstainThreshold);
    }
  } catch {
    // Graceful timeout fallback to Mode A
  }

  return predictHybrid(text, context, undefined, weights, abstainThreshold);
}
