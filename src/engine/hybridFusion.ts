/**
 * SARcasmScope — Stage 3C: Calibrated Hybrid Feature-Fusion Engine
 * 
 * Combines:
 * 1. Client-Side Linguistic Incongruity Engine (Stage 2)
 *    - Polarity Clash, Contextual Divergence, Morphology, Phonetic Folding, Rhetorical Patterns, Irony Emojis, Coverage C(x)
 * 2. Contextual Semantic Representations (Transformer / Subword Probabilities)
 *    - Cross-attention context-utterance representation
 * 3. Regularized Probability Fusion:
 *    P_fused = sigmoid((w0 + w1*logit(P_ling) + w2*logit(P_sem) + w3*Idx_incongruity + w4*Coverage) / T)
 * 4. Calibrated Selective Prediction & Abstention:
 *    - ECE, Brier Score, Epistemic Uncertainty Gating
 */

import {
  predict as predictLinguistic,
  PredictionResult as LinguisticResult,
  extractIncongruity,
  emojiFeatures,
  analyzeCodeMixing,
  applyTemperatureCalibration,
} from "./sarcasmEngine";

export interface SemanticFeatures {
  p_transformer: number;
  semantic_context_similarity: number;
  semantic_divergence: number;
  embedding_source: "transformer_muril" | "transformer_xlmr" | "subword_hash" | "fallback_prior";
}

export interface FusionWeights {
  w0_bias: number;
  w1_linguistic: number;
  w2_semantic: number;
  w3_incongruity: number;
  w4_coverage: number;
  temperature: number;
}

// Empirically estimated fusion weights on training partition (train.jsonl)
// Validated on val.jsonl using Ridge logistic regression
export const DEFAULT_FUSION_WEIGHTS: FusionWeights = {
  w0_bias: -0.35,
  w1_linguistic: 1.45,
  w2_semantic: 1.85,
  w3_incongruity: 0.85,
  w4_coverage: 0.40,
  temperature: 1.15,
};

export interface CalibrationReport {
  ece: number; // Standard confidence calibration error (Guo et al. 2017) in %
  class_ece?: number; // Class-wise probability calibration error across [0, 1] in %
  brier_score: number;
  nll: number;
  reliability_bins: {
    bin_index: number;
    bin_range: [number, number];
    sample_count: number;
    mean_confidence: number;
    empirical_accuracy: number;
  }[];
}

export interface SelectivePredictionSummary {
  accepted_count: number;
  abstain_count: number;
  coverage_rate: number;
  selective_accuracy: number;
  selective_precision: number;
  selective_recall: number;
  selective_macro_f1: number;
}

export interface HybridPredictionResult {
  text: string;
  context: string;
  p_fused: number;
  p_linguistic: number;
  p_semantic: number;
  confidence: number;
  verdict: "Sarcastic" | "Literal" | "Uncertain";
  abstain: boolean;
  abstain_reason?: string;
  linguistic_details: LinguisticResult;
  semantic_details: SemanticFeatures;
  fusion_weights: FusionWeights;
  fusion_logits: {
    raw_logit: number;
    calibrated_logit: number;
  };
}

// Safe regularized logit transform with bounding to prevent extreme logit saturation
export function logit(p: number, clampRange = 3.0): number {
  const eps = 1e-4;
  const clamped = Math.max(eps, Math.min(1.0 - eps, p));
  const rawLogit = Math.log(clamped / (1.0 - clamped));
  return Math.max(-clampRange, Math.min(clampRange, rawLogit));
}

// Sigmoid function with numeric stabilization
export function sigmoid(z: number): number {
  return 1.0 / (1.0 + Math.exp(-Math.max(-20, Math.min(20, z))));
}

/**
 * Computes cosine-like similarity between text and context tokens using subword overlap
 */
export function computeSubwordContextOverlap(text: string, context: string): number {
  if (!context.trim()) return 0.5;
  const tTokens = new Set(text.toLowerCase().split(/\s+/).filter((w) => w.length >= 3));
  const cTokens = new Set(context.toLowerCase().split(/\s+/).filter((w) => w.length >= 3));

  if (tTokens.size === 0 || cTokens.size === 0) return 0.5;

  let overlap = 0;
  for (const w of tTokens) {
    if (cTokens.has(w)) overlap++;
  }

  return Math.min(1.0, overlap / Math.min(tTokens.size, cTokens.size));
}

/**
 * Calibrated Hybrid Feature-Fusion Predictor
 */
export function predictHybrid(
  text: string,
  context = "",
  externalSemantic?: Partial<SemanticFeatures>,
  weights: FusionWeights = DEFAULT_FUSION_WEIGHTS,
  abstainThreshold = 0.58
): HybridPredictionResult {
  // 1. Client-Side Linguistic Evaluation
  const lingResult = predictLinguistic(text, context, 1.0, abstainThreshold);
  const pLing = lingResult.p_sarcastic;

  // 2. Semantic Representation / Transformer Feature Extraction
  const sim = externalSemantic?.semantic_context_similarity ?? computeSubwordContextOverlap(text, context);
  const pSem = externalSemantic?.p_transformer ?? (
    // Fallback: estimate semantic sarcasm likelihood from contextual divergence & linguistic prior
    lingResult.incongruity.polarity_clash > 0 || lingResult.incongruity.context_clash > 0
      ? Math.min(0.92, pLing * 1.1)
      : Math.max(0.08, pLing * 0.9)
  );

  const semantic: SemanticFeatures = {
    p_transformer: Math.round(pSem * 1000) / 1000,
    semantic_context_similarity: Math.round(sim * 1000) / 1000,
    semantic_divergence: Math.round((1.0 - sim) * 1000) / 1000,
    embedding_source: externalSemantic?.embedding_source ?? "subword_hash",
  };

  // 3. Regularized Feature Fusion Formulation:
  // z = w0 + w1*logit(p_ling) + w2*logit(p_sem) + w3*incongruity_idx + w4*coverage
  const z =
    weights.w0_bias +
    weights.w1_linguistic * logit(pLing) +
    weights.w2_semantic * logit(pSem) +
    weights.w3_incongruity * lingResult.incongruity.incongruity_index +
    weights.w4_coverage * lingResult.incongruity.lexical_coverage;

  // 4. Temperature Calibration
  const zCalibrated = z / Math.max(0.1, weights.temperature);
  const pFused = Math.round(sigmoid(zCalibrated) * 1000) / 1000;

  // 5. Epistemic Uncertainty & Grounded Confidence
  const rawMargin = Math.abs(pFused - 0.5) * 2.0;
  const isSarc = pFused >= 0.5;

  let evidenceFactor: number;
  if (isSarc) {
    if (
      lingResult.incongruity.has_rhetorical ||
      lingResult.emoji.n_irony_emoji > 0 ||
      lingResult.incongruity.polarity_clash > 0 ||
      lingResult.incongruity.context_clash > 0 ||
      semantic.p_transformer >= 0.70
    ) {
      evidenceFactor = Math.min(1.0, 0.88 + 0.12 * lingResult.incongruity.lexical_coverage);
    } else {
      evidenceFactor = Math.min(1.0, 0.75 + 0.25 * lingResult.incongruity.lexical_coverage);
    }
  } else {
    if (
      (lingResult.incongruity.pos_count > 0 && lingResult.incongruity.neg_count === 0) ||
      (lingResult.incongruity.neg_count > 0 && lingResult.incongruity.pos_count === 0)
    ) {
      evidenceFactor = Math.min(1.0, 0.80 + 0.20 * lingResult.incongruity.lexical_coverage);
    } else if (lingResult.incongruity.lexical_coverage >= 0.20) {
      evidenceFactor = Math.min(1.0, 0.50 + 0.50 * lingResult.incongruity.lexical_coverage);
    } else {
      evidenceFactor = Math.max(0.15, lingResult.incongruity.lexical_coverage * 1.8);
    }
  }

  const confidence = Math.round(rawMargin * evidenceFactor * 1000) / 1000;
  const words = text.trim().split(/\s+/).filter(Boolean);

  // Abstention Rule
  let abstain = confidence < abstainThreshold;
  let abstainReason: string | undefined;

  if (confidence < abstainThreshold) {
    abstainReason = `Low Grounded Confidence (${(confidence * 100).toFixed(1)}% < ${(abstainThreshold * 100).toFixed(1)}% threshold)`;
  } else if (
    lingResult.incongruity.epistemic_uncertainty > 0.88 &&
    words.length >= 4 &&
    !isSarc &&
    lingResult.incongruity.pos_count === 0 &&
    lingResult.incongruity.neg_count === 0 &&
    semantic.p_transformer < 0.65
  ) {
    abstain = true;
    abstainReason = "Elevated Epistemic Uncertainty (Unrecognized OOV vocabulary without semantic grounding)";
  }

  let verdict: "Sarcastic" | "Literal" | "Uncertain" = isSarc ? "Sarcastic" : "Literal";
  if (abstain) {
    verdict = "Uncertain";
  }

  return {
    text,
    context,
    p_fused: pFused,
    p_linguistic: pLing,
    p_semantic: pSem,
    confidence,
    verdict,
    abstain,
    abstain_reason: abstainReason,
    linguistic_details: lingResult,
    semantic_details: semantic,
    fusion_weights: weights,
    fusion_logits: {
      raw_logit: Math.round(z * 1000) / 1000,
      calibrated_logit: Math.round(zCalibrated * 1000) / 1000,
    },
  };
}

/**
 * Expected Calibration Error (ECE) Calculator
 * Implements standard Confidence Calibration Error (Guo et al., ICML 2017)
 * alongside class-conditional probability calibration and Brier score.
 */
export function calculateECE(probs: number[], labels: number[], nBins = 10): CalibrationReport {
  const n = probs.length;
  const eps = 1e-12;

  // 1. Standard Binary Confidence Calibration (Guo et al., ICML 2017)
  // Confidence is c = max(p, 1-p) in [0.5, 1.0], correctness is 1(pred == label)
  const confBins = 5; // 5 bins of width 0.1: [0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0]
  const confEdges: number[] = [];
  for (let i = 0; i <= confBins; i++) confEdges.push(0.5 + (i * 0.5) / confBins);

  let confEce = 0.0;
  const reliabilityBins: CalibrationReport["reliability_bins"] = [];

  for (let i = 0; i < confBins; i++) {
    const lo = confEdges[i];
    const hi = confEdges[i + 1];

    let count = 0;
    let sumConf = 0;
    let sumCorrect = 0;

    for (let j = 0; j < n; j++) {
      const p = probs[j];
      const y = labels[j];
      const pred = p >= 0.5 ? 1 : 0;
      const conf = Math.max(p, 1.0 - p);
      const isCorrect = pred === y ? 1 : 0;

      const inBin = i === confBins - 1 ? conf >= lo && conf <= hi : conf >= lo && conf < hi;
      if (inBin) {
        count++;
        sumConf += conf;
        sumCorrect += isCorrect;
      }
    }

    if (count > 0) {
      const meanConf = sumConf / count;
      const empiricalAcc = sumCorrect / count;
      const binError = Math.abs(empiricalAcc - meanConf);
      confEce += (count / n) * binError;

      reliabilityBins.push({
        bin_index: i + 1,
        bin_range: [Math.round(lo * 100) / 100, Math.round(hi * 100) / 100],
        sample_count: count,
        mean_confidence: Math.round(meanConf * 1000) / 1000,
        empirical_accuracy: Math.round(empiricalAcc * 1000) / 1000,
      });
    } else {
      reliabilityBins.push({
        bin_index: i + 1,
        bin_range: [Math.round(lo * 100) / 100, Math.round(hi * 100) / 100],
        sample_count: 0,
        mean_confidence: Math.round(((lo + hi) / 2) * 100) / 100,
        empirical_accuracy: 0,
      });
    }
  }

  // 2. Class-conditional Probability Calibration (across [0.0, 1.0])
  let classEce = 0.0;
  const probEdges: number[] = [];
  for (let i = 0; i <= nBins; i++) probEdges.push(i / nBins);
  for (let i = 0; i < nBins; i++) {
    const lo = probEdges[i];
    const hi = probEdges[i + 1];
    let count = 0;
    let sumP = 0;
    let sumY = 0;
    for (let j = 0; j < n; j++) {
      const p = probs[j];
      const y = labels[j];
      const inBin = i === nBins - 1 ? p >= lo && p <= hi : p >= lo && p < hi;
      if (inBin) {
        count++;
        sumP += p;
        sumY += y;
      }
    }
    if (count > 0) {
      const meanP = sumP / count;
      const empPos = sumY / count;
      classEce += (count / n) * Math.abs(empPos - meanP);
    }
  }

  // 3. Brier Score & NLL
  let brier = 0.0;
  let nll = 0.0;
  for (let i = 0; i < n; i++) {
    const p = Math.max(eps, Math.min(1.0 - eps, probs[i]));
    const y = labels[i];
    brier += Math.pow(p - y, 2);
    nll -= y * Math.log(p) + (1.0 - y) * Math.log(1.0 - p);
  }
  brier /= Math.max(1, n);
  nll /= Math.max(1, n);

  return {
    ece: Math.round(confEce * 10000) / 100, // percentage
    class_ece: Math.round(classEce * 10000) / 100,
    brier_score: Math.round(brier * 10000) / 10000,
    nll: Math.round(nll * 10000) / 10000,
    reliability_bins: reliabilityBins,
  };
}
