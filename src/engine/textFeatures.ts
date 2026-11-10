/**
 * SARcasmScope — Shared Feature Extraction Utilities
 *
 * Guarantees bit-for-bit identical tokenization, Indic stemming,
 * Romanized phonetic folding, and character n-gram extraction across
 * offline training/calibration (`eval/train_and_calibrate.ts`) and
 * runtime browser inference (`src/engine/statisticalModel.ts`).
 */

import {
  cleanText,
  stemIndicToken,
  normalizeRomanized,
  extractEmojis,
} from "./sarcasmEngine";

export interface TextFeatureOptions {
  useContext?: boolean;
  useMorph?: boolean;
  usePhonetic?: boolean;
  useUnigrams?: boolean;
  useBigrams?: boolean;
}

function rawWordTokens(text: string): string[] {
  return text
    .toLowerCase()
    .normalize("NFKC")
    .split(/\s+/)
    .map((w) => w.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ""))
    .filter((w) => w.length > 0);
}

/**
 * Extracts word 1-2 n-grams enriched with optional Indic stems (`st:`),
 * Romanized phonetic keys (`ph:`), emoji markers (`em:`), and context-prefixed
 * tokens (`cw:`). Avoids context-bigram bloat so top-8K vocabulary focuses on
 * high-signal utterance and context unigrams/bigrams.
 */
export function extractWordFeatures(
  text: string,
  context = "",
  opts: TextFeatureOptions = {}
): string[] {
  const useContext = opts.useContext ?? true;
  const useMorph = opts.useMorph ?? true;
  const usePhonetic = opts.usePhonetic ?? true;
  const useUnigrams = opts.useUnigrams ?? true;
  const useBigrams = opts.useBigrams ?? true;

  const feats: string[] = [];
  const textTokens = rawWordTokens(text);
  const textEmojis = extractEmojis(text);

  if (useUnigrams) {
    for (const w of textTokens) {
      feats.push(w);
      if (useMorph) {
        const { stem } = stemIndicToken(w);
        if (stem && stem !== w) {
          feats.push(`st:${stem}`);
        }
      }
      if (usePhonetic && /[a-z]/i.test(w)) {
        const { folded } = normalizeRomanized(w);
        if (folded && folded !== w) {
          feats.push(`ph:${folded}`);
        }
      }
    }
    for (const e of textEmojis) {
      feats.push(`em:${e}`);
    }
  }

  if (useBigrams && textTokens.length >= 2) {
    for (let i = 0; i < textTokens.length - 1; i++) {
      feats.push(`bg:${textTokens[i]}_${textTokens[i + 1]}`);
    }
  }

  if (useContext && context.trim()) {
    const ctxTokens = rawWordTokens(context).slice(-80);
    if (useUnigrams) {
      for (const cw of ctxTokens) {
        feats.push(`cw:${cw}`);
        if (useMorph) {
          const { stem } = stemIndicToken(cw);
          if (stem && stem !== cw) {
            feats.push(`cw_st:${stem}`);
          }
        }
      }
    }
  }

  return feats;
}

/**
 * Extracts character 3-5 n-grams across boundary-marked utterance
 * and optional recent conversational context.
 */
export function extractCharFeatures(
  text: string,
  context = "",
  useContext = true
): string[] {
  const normText = text.toLowerCase().normalize("NFKC").replace(/\s+/g, " ").trim();
  const normCtx =
    useContext && context.trim()
      ? context.toLowerCase().normalize("NFKC").replace(/\s+/g, " ").trim().slice(-240)
      : "";

  const combined = normCtx ? `^${normText}$|^${normCtx}$` : `^${normText}$`;
  const capped = combined.slice(0, 520);
  const feats: string[] = [];

  for (let n = 3; n <= 5; n++) {
    for (let i = 0; i <= capped.length - n; i++) {
      feats.push(capped.substring(i, i + n));
    }
  }

  return feats;
}

/**
 * Clean helper re-exported for downstream diagnostics.
 */
export function normalizeTextClean(text: string): string {
  return cleanText(text);
}

/**
 * Safe regularized log-odds transform preventing infinite logits.
 */
export function boundedLogit(p: number, clampRange = 4.0): number {
  const eps = 1e-4;
  const clamped = Math.max(eps, Math.min(1.0 - eps, p));
  const raw = Math.log(clamped / (1.0 - clamped));
  return Math.max(-clampRange, Math.min(clampRange, raw));
}

export function stableSigmoid(z: number): number {
  return 1.0 / (1.0 + Math.exp(-Math.max(-20, Math.min(20, z))));
}

/**
 * Deterministic 32-bit FNV-1a hash + Mulberry32 PRNG for reproducible tests/perturbations.
 */
export function hashString32(str: string): number {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function createSeededPRNG(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
