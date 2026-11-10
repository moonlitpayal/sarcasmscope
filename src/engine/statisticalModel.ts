/**
 * SARcasmScope — Runtime Level-1 Statistical Base Experts (`WordLex` & `CharSub`)
 *
 * Loads deterministic sparse TF-IDF Logistic Regression parameters fitted exclusively
 * on `train.jsonl` (via `eval/train_and_calibrate.ts` and serialized in
 * `src/engine/trainedModelData.ts`). Supports modular feature toggles so true
 * controlled ablations (morphological stemming, phonetic folding, word n-grams,
 * character 3-5 grams, conversational context) execute directly through the real runtime.
 */

import { TRAINED_MODEL_ARTIFACT, SerializedSparseModel } from "./trainedModelData";
import {
  extractWordFeatures,
  extractCharFeatures,
  boundedLogit,
  stableSigmoid,
} from "./textFeatures";

export interface StatisticalOptions {
  useWord?: boolean;
  useChar?: boolean;
  useContext?: boolean;
  useMorph?: boolean;
  usePhonetic?: boolean;
  useUnigrams?: boolean;
  useBigrams?: boolean;
}

export interface StatisticalPrediction {
  p_word: number;
  p_char: number;
  p_statistical: number;
  word_matches: number;
  char_matches: number;
  context_similarity: number;
}

const wordVocabMap = new Map<string, number>();
TRAINED_MODEL_ARTIFACT.wordModel.vocab.forEach((term, idx) => {
  wordVocabMap.set(term, idx);
});

const charVocabMap = new Map<string, number>();
TRAINED_MODEL_ARTIFACT.charModel.vocab.forEach((term, idx) => {
  charVocabMap.set(term, idx);
});

function scoreSparse(
  tokens: string[],
  vocabMap: Map<string, number>,
  model: SerializedSparseModel
): { prob: number; matches: number } {
  const counts = new Map<number, number>();
  for (let i = 0; i < tokens.length; i++) {
    const idx = vocabMap.get(tokens[i]);
    if (idx !== undefined) {
      counts.set(idx, (counts.get(idx) || 0) + 1);
    }
  }

  if (counts.size === 0) {
    return { prob: stableSigmoid(model.bias), matches: 0 };
  }

  const idf = model.idf;
  const weights = model.weights;
  const idxs: number[] = [];
  const vals: number[] = [];
  let normSq = 0.0;

  for (const [idx, tf] of counts.entries()) {
    const val = (1.0 + Math.log(tf)) * idf[idx];
    idxs.push(idx);
    vals.push(val);
    normSq += val * val;
  }

  let dot = model.bias;
  if (normSq > 0.0) {
    const invNorm = 1.0 / Math.sqrt(normSq);
    for (let k = 0; k < idxs.length; k++) {
      dot += weights[idxs[k]] * (vals[k] * invNorm);
    }
  }

  return { prob: stableSigmoid(dot), matches: counts.size };
}

/**
 * Computes cosine similarity between subword / token sets of utterance and context.
 */
export function computeStatisticalOverlap(text: string, context: string): number {
  if (!context || !context.trim()) return 0.5;
  const tToks = new Set(
    text
      .toLowerCase()
      .split(/\s+/)
      .map((w) => w.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ""))
      .filter((w) => w.length >= 3)
  );
  const cToks = new Set(
    context
      .toLowerCase()
      .split(/\s+/)
      .map((w) => w.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ""))
      .filter((w) => w.length >= 3)
  );

  if (tToks.size === 0 || cToks.size === 0) return 0.5;

  let overlap = 0;
  for (const w of tToks) {
    if (cToks.has(w)) overlap++;
  }
  return Math.min(1.0, overlap / Math.min(tToks.size, cToks.size));
}

export function scoreWordModel(
  text: string,
  context = "",
  opts: StatisticalOptions = {}
): { prob: number; matches: number } {
  const tokens = extractWordFeatures(text, context, {
    useContext: opts.useContext ?? true,
    useMorph: opts.useMorph ?? true,
    usePhonetic: opts.usePhonetic ?? true,
    useUnigrams: opts.useUnigrams ?? true,
    useBigrams: opts.useBigrams ?? true,
  });
  return scoreSparse(tokens, wordVocabMap, TRAINED_MODEL_ARTIFACT.wordModel);
}

export function scoreCharModel(
  text: string,
  context = "",
  opts: StatisticalOptions = {}
): { prob: number; matches: number } {
  const grams = extractCharFeatures(text, context, opts.useContext ?? true);
  return scoreSparse(grams, charVocabMap, TRAINED_MODEL_ARTIFACT.charModel);
}

/**
 * Computes Level-1 base expert probabilities (WordLex + CharSub).
 */
export function predictStatistical(
  text: string,
  context = "",
  opts: StatisticalOptions = {}
): StatisticalPrediction {
  const useWord = opts.useWord ?? true;
  const useChar = opts.useChar ?? true;
  const useContext = opts.useContext ?? true;

  const wordRes = useWord
    ? scoreWordModel(text, context, opts)
    : { prob: 0.5, matches: 0 };

  const charRes = useChar
    ? scoreCharModel(text, context, opts)
    : { prob: 0.5, matches: 0 };

  let pEnsemble = 0.5;
  if (useWord && useChar) {
    pEnsemble = stableSigmoid(
      0.5 * (boundedLogit(wordRes.prob, 4.0) + boundedLogit(charRes.prob, 4.0))
    );
  } else if (useWord) {
    pEnsemble = wordRes.prob;
  } else if (useChar) {
    pEnsemble = charRes.prob;
  }

  const contextSim = computeStatisticalOverlap(text, useContext ? context : "");

  return {
    p_word: Math.round(wordRes.prob * 1000) / 1000,
    p_char: Math.round(charRes.prob * 1000) / 1000,
    p_statistical: Math.round(pEnsemble * 1000) / 1000,
    word_matches: wordRes.matches,
    char_matches: charRes.matches,
    context_similarity: Math.round(contextSim * 1000) / 1000,
  };
}
