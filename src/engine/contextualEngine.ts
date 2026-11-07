/**
 * SARcasmScope — Stage 3E: Hybrid Contextual Engine Adapter
 * 
 * Supports two operational deployment modes:
 * Mode A: Standalone Client-Side
 *   - Runs Stage 2 linguistic incongruity + subword hash representations directly in-browser
 *   - Zero server dependency, latency < 10ms, zero marginal cloud cost
 * 
 * Mode B: Hybrid API Inference
 *   - Calls `/api/predict-contextual` for server-side transformer inference (MuRIL / XLM-R)
 *   - Fuses deep transformer probabilities with client-side linguistic features
 *   - Graceful timeout fallback: If API fails, times out, or is offline,
 *     it automatically reverts to Mode A with zero UI disruption
 */

import { predictHybrid, HybridPredictionResult, SemanticFeatures } from "./hybridFusion";

export interface ContextualPredictionOptions {
  mode?: "standalone" | "hybrid";
  apiUrl?: string;
  timeoutMs?: number;
  abstainThreshold?: number;
}

export interface ApiTransformerResponse {
  p_sarcastic: number;
  model_name: string;
  latency_ms: number;
  status: "success" | "error";
}

/**
 * Predicts sarcasm using either standalone hybrid fusion or remote contextual API
 */
export async function predictContextual(
  text: string,
  context = "",
  options: ContextualPredictionOptions = {}
): Promise<HybridPredictionResult> {
  const mode = options.mode ?? "standalone";
  const apiUrl = options.apiUrl ?? "/api/predict-contextual";
  const timeoutMs = options.timeoutMs ?? 1500;
  const abstainThreshold = options.abstainThreshold ?? 0.58;

  // Mode A: Immediate standalone inference
  if (mode === "standalone" || typeof window === "undefined") {
    return predictHybrid(text, context, undefined, undefined, abstainThreshold);
  }

  // Mode B: Hybrid API inference with graceful timeout fallback
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
      return predictHybrid(text, context, externalSemantic, undefined, abstainThreshold);
    }
  } catch (err) {
    // Timeout or network error -> Gracefully fall back to standalone client mode
    // (Anti-fragile design: zero crash, zero disruption)
  }

  return predictHybrid(text, context, undefined, undefined, abstainThreshold);
}
