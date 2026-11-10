/**
 * SARcasmScope — Calibrated 3-Stage Hybrid Feature-Fusion Engine
 *
 * Level 1 Base Experts (fitted strictly on train.jsonl, N = 3,373):
 *   1. Linguistic Incongruity Engine (Stage 2 symbolic rules, Indic stemming, Romanized phonetic folding, C(x))
 *   2. WordLex (word 1-2 L2-normalized sublinear TF-IDF Logistic Regression w/ stems, phonetic keys & cw: context)
 *   3. CharSub (character 3-5 L2-normalized sublinear TF-IDF Logistic Regression)
 *
 * Level 2 Meta-Learner (fitted strictly on 5-fold group-stratified train OOF predictions):
 *   z_raw = w0 + w1*logit(p_word, 4) + w2*logit(p_char, 4) + w3*logit(p_ling, 3) + w4*incongruity_index + w5*lexical_coverage
 *
 * Level 3 Post-Hoc Calibration & Selective Abstention (fitted strictly on val.jsonl, N = 1,104):
 *   P_fused = sigmoid(z_raw / T*)
 *   Confidence = 2 * |P_fused - 0.5|, Abstain iff Confidence < tau*
 */

import {
  predict as predictLinguistic,
  PredictionResult as LinguisticResult,
  OcclusionResult,
  TokenAttribution,
  RobustnessResult,
  PerturbationRow,
  POSITIVE_CUES,
  NEGATIVE_CUES,
  INTENSIFIERS,
  stemIndicToken,
  normalizeRomanized,
  stripEmojis,
} from "./sarcasmEngine";
import { TRAINED_MODEL_ARTIFACT } from "./trainedModelData";
import { predictStatistical, computeStatisticalOverlap } from "./statisticalModel";
import {
  boundedLogit,
  stableSigmoid,
  hashString32,
  createSeededPRNG,
} from "./textFeatures";

export interface SemanticFeatures {
  p_transformer: number;
  p_word: number;
  p_char: number;
  semantic_context_similarity: number;
  semantic_divergence: number;
  embedding_source:
    | "statistical_oof_stack"
    | "transformer_muril"
    | "transformer_xlmr"
    | "fallback_prior";
}

export interface FusionWeights {
  w0_bias: number;
  w1_word: number;
  w2_char: number;
  w3_linguistic: number;
  w4_incongruity: number;
  w5_coverage: number;
  temperature: number;
}

export interface HybridAblationOptions {
  useWord?: boolean;
  useChar?: boolean;
  useLinguistic?: boolean;
  useIncongruity?: boolean;
  useCoverage?: boolean;
  useContext?: boolean;
  useMorph?: boolean;
  usePhonetic?: boolean;
  useUnigrams?: boolean;
  useBigrams?: boolean;
  useCalibration?: boolean;
}

export const TRAINED_TEMPERATURE: number = TRAINED_MODEL_ARTIFACT.fusionWeights.temperature;
export const DEFAULT_ABSTAIN_THRESHOLD: number = TRAINED_MODEL_ARTIFACT.defaultAbstainThreshold;

export const DEFAULT_FUSION_WEIGHTS: FusionWeights = {
  ...TRAINED_MODEL_ARTIFACT.fusionWeights,
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
  p_word: number;
  p_char: number;
  p_semantic: number;
  confidence: number;
  threshold: number;
  verdict: "Sarcastic" | "Literal" | "Uncertain";
  abstain: boolean;
  abstain_reason?: string;
  incongruity: LinguisticResult["incongruity"];
  language: LinguisticResult["language"];
  emoji: LinguisticResult["emoji"];
  linguistic_details: LinguisticResult;
  semantic_details: SemanticFeatures;
  fusion_weights: FusionWeights;
  fusion_logits: {
    raw_logit: number;
    calibrated_logit: number;
  };
}

export function logit(p: number, clampRange = 4.0): number {
  return boundedLogit(p, clampRange);
}

export function sigmoid(z: number): number {
  return stableSigmoid(z);
}

export function computeSubwordContextOverlap(text: string, context: string): number {
  return computeStatisticalOverlap(text, context);
}

/**
 * Calibrated Hybrid Feature-Fusion Predictor
 */
export function predictHybrid(
  text: string,
  context = "",
  externalSemantic?: Partial<SemanticFeatures>,
  weights: FusionWeights = DEFAULT_FUSION_WEIGHTS,
  abstainThreshold = DEFAULT_ABSTAIN_THRESHOLD,
  ablation: HybridAblationOptions = {}
): HybridPredictionResult {
  const useWord = ablation.useWord ?? true;
  const useChar = ablation.useChar ?? true;
  const useLinguistic = ablation.useLinguistic ?? true;
  const useIncongruity = ablation.useIncongruity ?? true;
  const useCoverage = ablation.useCoverage ?? true;
  const useContext = ablation.useContext ?? true;
  const useMorph = ablation.useMorph ?? true;
  const usePhonetic = ablation.usePhonetic ?? true;
  const useUnigrams = ablation.useUnigrams ?? true;
  const useBigrams = ablation.useBigrams ?? true;
  const useCalibration = ablation.useCalibration ?? true;

  const effContext = useContext ? context : "";

  // 1. Level-1 Client Linguistic Analysis (always extracted for UI metadata, masked from logit if ablated)
  const lingResult = predictLinguistic(text, effContext, 1.0, abstainThreshold, {
    useMorph,
    usePhonetic,
  });
  const pLing = useLinguistic ? lingResult.p_sarcastic : 0.5;
  const effIncIdx = useLinguistic && useIncongruity ? lingResult.incongruity.incongruity_index : 0.0;
  const effCov = useLinguistic && useCoverage ? lingResult.incongruity.lexical_coverage : 0.0;

  // 2. Level-1 Statistical / Semantic Base Experts (WordLex & CharSub, or Mode B external transformer)
  const statPred = predictStatistical(text, effContext, {
    useWord,
    useChar,
    useContext,
    useMorph,
    usePhonetic,
    useUnigrams,
    useBigrams,
  });

  const pWord = externalSemantic?.p_word ?? externalSemantic?.p_transformer ?? statPred.p_word;
  const pChar = externalSemantic?.p_char ?? externalSemantic?.p_transformer ?? statPred.p_char;
  const pSem = externalSemantic?.p_transformer ?? statPred.p_statistical;
  const sim = externalSemantic?.semantic_context_similarity ?? statPred.context_similarity;

  const semantic: SemanticFeatures = {
    p_transformer: Math.round(pSem * 1000) / 1000,
    p_word: Math.round(pWord * 1000) / 1000,
    p_char: Math.round(pChar * 1000) / 1000,
    semantic_context_similarity: Math.round(sim * 1000) / 1000,
    semantic_divergence: Math.round((1.0 - sim) * 1000) / 1000,
    embedding_source: externalSemantic?.embedding_source ?? "statistical_oof_stack",
  };

  // 3. Level-2 Regularized Ridge Meta-Learner Formulation
  const z =
    weights.w0_bias +
    (useWord ? weights.w1_word * boundedLogit(pWord, 4.0) : 0.0) +
    (useChar ? weights.w2_char * boundedLogit(pChar, 4.0) : 0.0) +
    (useLinguistic ? weights.w3_linguistic * boundedLogit(pLing, 3.0) : 0.0) +
    weights.w4_incongruity * effIncIdx +
    weights.w5_coverage * effCov;

  // 4. Level-3 Post-Hoc Validation Temperature Scaling
  const effT = useCalibration ? Math.max(0.1, weights.temperature) : 1.0;
  const zCalibrated = z / effT;
  const pFused = Math.round(stableSigmoid(zCalibrated) * 1000) / 1000;

  // 5. Calibrated Confidence & Selective Abstention Gate
  // Strictly matches Confidence = 2 * |P_calibrated - 0.5|
  const confidence = Math.round(Math.abs(pFused - 0.5) * 2.0 * 1000) / 1000;
  const isSarc = pFused >= 0.5;
  const abstain = confidence < abstainThreshold;
  const abstainReason = abstain
    ? `Calibrated Confidence (${(confidence * 100).toFixed(1)}% < ${(abstainThreshold * 100).toFixed(1)}% threshold)`
    : undefined;

  const verdict: "Sarcastic" | "Literal" | "Uncertain" = abstain
    ? "Uncertain"
    : isSarc
      ? "Sarcastic"
      : "Literal";

  return {
    text,
    context: effContext,
    p_fused: pFused,
    p_linguistic: pLing,
    p_word: semantic.p_word,
    p_char: semantic.p_char,
    p_semantic: semantic.p_transformer,
    confidence,
    threshold: abstainThreshold,
    verdict,
    abstain,
    abstain_reason: abstainReason,
    incongruity: lingResult.incongruity,
    language: lingResult.language,
    emoji: lingResult.emoji,
    linguistic_details: lingResult,
    semantic_details: semantic,
    fusion_weights: { ...weights, temperature: effT },
    fusion_logits: {
      raw_logit: Math.round(z * 1000) / 1000,
      calibrated_logit: Math.round(zCalibrated * 1000) / 1000,
    },
  };
}

/**
 * Index-based Leave-One-Token-Out (LOTO) Occlusion Attribution for the Hybrid Fusion Engine.
 * Correctly handles repeated tokens by removing index `i` rather than all identical strings.
 */
export function computeHybridOcclusion(
  text: string,
  context = "",
  temperature = TRAINED_TEMPERATURE,
  abstainThreshold = DEFAULT_ABSTAIN_THRESHOLD,
  topK = 5
): OcclusionResult {
  const weights: FusionWeights = { ...DEFAULT_FUSION_WEIGHTS, temperature };
  const tokens = text.trim().split(/\s+/).filter(Boolean);
  const baseResult = predictHybrid(text, context, undefined, weights, abstainThreshold);
  const baseP = baseResult.p_fused;

  if (tokens.length === 0) {
    return {
      p_sarcastic: baseP,
      attributions: [],
      top_sarcastic: [],
      top_literal: [],
    };
  }

  const perTokenData = tokens.map((token, i) => {
    const occludedText = [...tokens.slice(0, i), ...tokens.slice(i + 1)].join(" ");
    const occRes = predictHybrid(occludedText, context, undefined, weights, abstainThreshold);
    const probDelta = Math.round((baseP - occRes.p_fused) * 1000) / 1000;

    const clean = token.toLowerCase().replace(/^[\p{P}\p{S}]+|[\p{P}\p{S}]+$/gu, "");
    const { stem: indicStem } = stemIndicToken(clean);
    const { folded: romanFolded } = normalizeRomanized(clean);

    let category: TokenAttribution["category"] = "neutral";
    if (POSITIVE_CUES.has(clean) || POSITIVE_CUES.has(indicStem) || POSITIVE_CUES.has(romanFolded)) {
      category = "positive";
    } else if (NEGATIVE_CUES.has(clean) || NEGATIVE_CUES.has(indicStem) || NEGATIVE_CUES.has(romanFolded)) {
      category = "negative";
    } else if (INTENSIFIERS.has(clean) || INTENSIFIERS.has(indicStem) || INTENSIFIERS.has(romanFolded)) {
      category = "intensifier";
    }

    return {
      token,
      clean_token: clean,
      category,
      probDelta,
      occLogit: occRes.fusion_logits.calibrated_logit,
    };
  });

  const allProbZero = perTokenData.every((d) => d.probDelta === 0);
  const nearSaturation = baseP >= 0.995 || baseP <= 0.005;
  const baseLogit = baseResult.fusion_logits.calibrated_logit;

  const attributions: TokenAttribution[] = perTokenData.map((d) => {
    const delta =
      allProbZero && nearSaturation && Math.abs(baseLogit - d.occLogit) > 1e-6
        ? Math.round(Math.max(-0.5, Math.min(0.5, (baseLogit - d.occLogit) * 0.25)) * 1000) / 1000
        : d.probDelta;
    return {
      token: d.token,
      delta,
      clean_token: d.clean_token,
      category: d.category,
    };
  });

  const ranked = [...attributions].sort((a, b) => b.delta - a.delta);
  const topSarcastic = ranked.filter((a) => a.delta > 0).slice(0, topK);
  const topLiteral = [...ranked].reverse().filter((a) => a.delta < 0).slice(0, topK);

  return {
    p_sarcastic: baseP,
    attributions,
    top_sarcastic: topSarcastic,
    top_literal: topLiteral,
  };
}

function graphemeClusters(word: string): string[] {
  if (typeof Intl !== "undefined" && (Intl as any).Segmenter) {
    const segmenter = new (Intl as any).Segmenter("und", { granularity: "grapheme" });
    return Array.from(segmenter.segment(word)).map((s: any) => s.segment);
  }
  return Array.from(word);
}

/**
 * Deterministic Hybrid Perturbation & Noise Stress Suite (seeded by input hash).
 */
export function perturbHybridAll(
  text: string,
  context = "",
  temperature = TRAINED_TEMPERATURE
): RobustnessResult {
  const weights: FusionWeights = { ...DEFAULT_FUSION_WEIGHTS, temperature };
  const baseResult = predictHybrid(text, context, undefined, weights);
  const baseP = baseResult.p_fused;
  const baseIsSarc = baseP >= 0.5;
  const rng = createSeededPRNG(hashString32(text));

  const words = text.split(/\s+/).filter(Boolean);

  // 1. Cluster-safe character noise
  const textCharNoise = words
    .map((w) => {
      const cl = graphemeClusters(w);
      if (cl.length >= 4) {
        const k = Math.floor(1 + rng() * (cl.length - 2));
        const op = rng();
        if (op < 0.33) {
          const tmp = cl[k];
          cl[k] = cl[k + 1];
          cl[k + 1] = tmp;
        } else if (op < 0.66) {
          cl.splice(k, 0, cl[k]);
        } else {
          cl.splice(k, 1);
        }
        return cl.join("");
      }
      return w;
    })
    .join(" ");

  // 2. Repeated letters / elongation
  const textRepeat = words
    .map((w) => {
      const cl = graphemeClusters(w);
      if (cl.length >= 3 && rng() < 0.6) {
        return w + cl[cl.length - 1].repeat(3);
      }
      return w;
    })
    .join(" ");

  // 3. Remove emojis
  const textNoEmoji = stripEmojis(text);

  // 4. Strip punctuation
  const textNoPunct = text.replace(/[\p{P}\p{S}]/gu, " ").replace(/\s+/g, " ").trim();

  // 5. Drop hashtags & mentions
  const textDropTags = words.filter((w) => !w.startsWith("#") && !w.startsWith("@")).join(" ");

  // 6. Truncate to first 6 words
  const textTruncate = words.slice(0, Math.min(6, words.length)).join(" ");

  // 7. Deterministic Fisher-Yates shuffle
  const shuffled = [...words];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const textShuffle = shuffled.join(" ");

  // 8. Uppercase
  const textUppercase = text.toUpperCase();

  const perturbations = [
    { name: "char_noise", desc: "Cluster-safe character swap/duplication", text: textCharNoise || text },
    { name: "repeat_letters", desc: "Elongated letters (e.g. soo, wahhh)", text: textRepeat || text },
    { name: "remove_emoji", desc: "Strip all irony & emotion emojis", text: textNoEmoji || text },
    { name: "strip_punct", desc: "Remove exclamation & question marks", text: textNoPunct || text },
    { name: "drop_tags", desc: "Remove hashtags and @mentions", text: textDropTags || text },
    { name: "truncate_6_words", desc: "Truncate text to first 6 words", text: textTruncate || text },
    { name: "shuffle_words", desc: "Deterministically permute word order", text: textShuffle || text },
    { name: "uppercase", desc: "Transform into full UPPERCASE", text: textUppercase || text },
  ];

  let retainedCount = 0;
  const rows: PerturbationRow[] = perturbations.map((item) => {
    const res = predictHybrid(item.text, context, undefined, weights);
    const pPert = res.p_fused;
    const pertIsSarc = pPert >= 0.5;
    const flipped = pertIsSarc !== baseIsSarc;
    if (!flipped) retainedCount++;
    return {
      perturbation: item.name,
      description: item.desc,
      perturbed_text: item.text,
      p_sarcastic: pPert,
      flipped,
      delta: Math.round((pPert - baseP) * 1000) / 1000,
    };
  });

  return {
    baseline_p_sarcastic: baseP,
    baseline_label: baseIsSarc ? "sarcastic" : "not sarcastic",
    results: rows,
    stability_score: Math.round((retainedCount / rows.length) * 100),
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

  const confBins = 5; // [0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0]
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
      confEce += (count / n) * Math.abs(empiricalAcc - meanConf);

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
      classEce += (count / n) * Math.abs(sumY / count - sumP / count);
    }
  }

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
    ece: Math.round(confEce * 10000) / 100,
    class_ece: Math.round(classEce * 10000) / 100,
    brier_score: Math.round(brier * 10000) / 10000,
    nll: Math.round(nll * 10000) / 10000,
    reliability_bins: reliabilityBins,
  };
}
