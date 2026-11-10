/**
 * SARcasmScope — Genuine 3-Stage Calibrated Feature-Fusion Trainer
 *
 * Strictly enforces split separation (test.jsonl is NEVER accessed here):
 *  - Level 1 Base Experts (fitted ONLY on train.jsonl, N = 3,373):
 *      * WordLex: Word 1-2 n-grams + Indic stems + Romanized phonetic keys + `cw:` context prefix
 *        L2-normalized sublinear TF-IDF Logistic Regression (full-batch Adam + L2 reg)
 *      * CharSub: Char 3-5 n-grams over utterance + immediate context
 *        L2-normalized sublinear TF-IDF Logistic Regression (full-batch Adam + L2 reg)
 *      * LinguisticEngine: Deterministic Stage 2 symbolic & incongruity features
 *  - 5-Fold Group-Stratified Cross-Validation inside train.jsonl (grouped by dataset__conversation_id):
 *      Generates strict out-of-fold probabilities [p_word_oof, p_char_oof]
 *  - Level 2 Meta-Learner (fitted ONLY on train.jsonl OOF + linguistic features):
 *      MAP-Regularized Ridge Logistic Regression with Gaussian prior N(mu_prior, lambda^-1 I)
 *      minimizing train-OOF binary cross-entropy + (lambda/2) * ||w - mu_prior||^2
 *  - Level 3 Post-Hoc Calibration & Selective Risk-Coverage (fitted ONLY on val.jsonl, N = 1,104):
 *      Optimizes scalar temperature T* minimizing validation NLL and profiles Risk-Coverage across tau
 *
 * Emits:
 *  - src/engine/trainedModelData.ts (serialized vocab, IDF, Level-1 weights, Level-2 meta-weights, T*, tau*)
 *  - eval_results/independent/training_calibration_summary.json
 *  - eval_results/independent/stage3_system_comparison.csv (Validation split comparison)
 *  - eval_results/independent/stage3b_representations_comparison.csv
 *  - eval_results/independent/stage3_evaluation_report.json
 */

import * as fs from "fs";
import * as path from "path";
import { predict as predictLinguistic } from "../src/engine/sarcasmEngine";
import {
  extractWordFeatures,
  extractCharFeatures,
  boundedLogit,
  stableSigmoid,
  createSeededPRNG,
} from "../src/engine/textFeatures";

interface DataRecord {
  id: string;
  dataset: string;
  language: string;
  text: string;
  context: string;
  conversation_id?: string;
  binary_sarcasm_label: 0 | 1;
}

interface SparseExample {
  indices: Int32Array;
  values: Float32Array;
  label: number;
}

interface SparseModelArtifact {
  vocab: string[];
  idf: number[];
  weights: number[];
  bias: number;
}

function loadJsonl(filepath: string): DataRecord[] {
  const lines = fs.readFileSync(filepath, "utf8").trim().split("\n");
  return lines.map((line) => JSON.parse(line));
}

function round5(x: number): number {
  return Math.round(x * 1e5) / 1e5;
}

function round4(x: number): number {
  return Math.round(x * 1e4) / 1e4;
}

function round2(x: number): number {
  return Math.round(x * 1e2) / 1e2;
}

/**
 * Fits an L2-normalized sublinear TF-IDF Logistic Regression model via full-batch Adam.
 */
function trainSparseLogisticModel(
  trainRecords: DataRecord[],
  extractor: (r: DataRecord) => string[],
  opts: {
    minDf: number;
    maxFeatures: number;
    l2: number;
    lr: number;
    epochs: number;
  }
): {
  model: SparseModelArtifact;
  predictRecord: (r: DataRecord) => number;
} {
  const docFreq = new Map<string, number>();
  const tokenized = trainRecords.map((r) => extractor(r));

  for (const tokens of tokenized) {
    const seen = new Set(tokens);
    for (const t of seen) {
      docFreq.set(t, (docFreq.get(t) || 0) + 1);
    }
  }

  const candidates = Array.from(docFreq.entries())
    .filter(([, df]) => df >= opts.minDf)
    .sort((a, b) => (b[1] !== a[1] ? b[1] - a[1] : a[0].localeCompare(b[0])))
    .slice(0, opts.maxFeatures);

  const vocabList = candidates.map(([t]) => t);
  const vocabMap = new Map<string, number>();
  vocabList.forEach((t, idx) => vocabMap.set(t, idx));

  const N = trainRecords.length;
  const V = vocabList.length;
  const idf = new Float64Array(V);
  for (let i = 0; i < V; i++) {
    const df = candidates[i][1];
    idf[i] = Math.log((N + 1) / (df + 1)) + 1.0;
  }

  const encode = (tokens: string[], label: number): SparseExample => {
    const counts = new Map<number, number>();
    for (const t of tokens) {
      const idx = vocabMap.get(t);
      if (idx !== undefined) counts.set(idx, (counts.get(idx) || 0) + 1);
    }
    const size = counts.size;
    const indices = new Int32Array(size);
    const values = new Float32Array(size);
    let k = 0;
    let normSq = 0;
    for (const [idx, tf] of counts.entries()) {
      const val = (1.0 + Math.log(tf)) * idf[idx];
      indices[k] = idx;
      values[k] = val;
      normSq += val * val;
      k++;
    }
    if (normSq > 0) {
      const invNorm = 1.0 / Math.sqrt(normSq);
      for (let j = 0; j < size; j++) values[j] *= invNorm;
    }
    return { indices, values, label };
  };

  const examples = trainRecords.map((r, i) => encode(tokenized[i], r.binary_sarcasm_label));

  const weights = new Float64Array(V);
  const mW = new Float64Array(V);
  const vW = new Float64Array(V);
  let bias = 0.0;
  let mB = 0.0;
  let vB = 0.0;
  const gradW = new Float64Array(V);

  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const invN = 1.0 / Math.max(1, N);

  for (let ep = 1; ep <= opts.epochs; ep++) {
    gradW.fill(0);
    let gradB = 0.0;

    for (let i = 0; i < N; i++) {
      const ex = examples[i];
      let dot = bias;
      for (let j = 0; j < ex.indices.length; j++) {
        dot += weights[ex.indices[j]] * ex.values[j];
      }
      const p = stableSigmoid(dot);
      const err = (p - ex.label) * invN;
      gradB += err;
      for (let j = 0; j < ex.indices.length; j++) {
        gradW[ex.indices[j]] += err * ex.values[j];
      }
    }

    const bc1 = 1.0 - Math.pow(beta1, ep);
    const bc2 = 1.0 - Math.pow(beta2, ep);

    mB = beta1 * mB + (1.0 - beta1) * gradB;
    vB = beta2 * vB + (1.0 - beta2) * gradB * gradB;
    bias -= opts.lr * (mB / bc1) / (Math.sqrt(vB / bc2) + eps);

    for (let j = 0; j < V; j++) {
      const g = gradW[j] + opts.l2 * weights[j];
      mW[j] = beta1 * mW[j] + (1.0 - beta1) * g;
      vW[j] = beta2 * vW[j] + (1.0 - beta2) * g * g;
      weights[j] -= opts.lr * (mW[j] / bc1) / (Math.sqrt(vW[j] / bc2) + eps);
    }
  }

  const predictRecord = (r: DataRecord): number => {
    const tokens = extractor(r);
    const ex = encode(tokens, r.binary_sarcasm_label);
    let dot = bias;
    for (let j = 0; j < ex.indices.length; j++) {
      dot += weights[ex.indices[j]] * ex.values[j];
    }
    return stableSigmoid(dot);
  };

  return {
    model: {
      vocab: vocabList,
      idf: Array.from(idf, (x) => round5(x)),
      weights: Array.from(weights, (x) => round5(x)),
      bias: round5(bias),
    },
    predictRecord,
  };
}

/**
 * Builds 5-fold group-stratified CV fold indices inside trainRecords.
 * Conversations (dataset__conversation_id) stay atomic across folds.
 */
function assignGroupStratifiedFolds(records: DataRecord[], kFolds = 5, seed = 42): Int32Array {
  const rng = createSeededPRNG(seed);
  const foldAssign = new Int32Array(records.length);

  const convGroups = new Map<string, number[]>();
  for (let i = 0; i < records.length; i++) {
    const r = records[i];
    const gKey = `${r.dataset}__${r.conversation_id || r.id}`;
    if (!convGroups.has(gKey)) convGroups.set(gKey, []);
    convGroups.get(gKey)!.push(i);
  }

  const strata = new Map<string, number[][]>();
  for (const idxs of convGroups.values()) {
    const first = records[idxs[0]];
    const hasSarc = idxs.some((i) => records[i].binary_sarcasm_label === 1) ? 1 : 0;
    const sKey = `${first.dataset}__${hasSarc}`;
    if (!strata.has(sKey)) strata.set(sKey, []);
    strata.get(sKey)!.push(idxs);
  }

  for (const groups of strata.values()) {
    const shuffled = [...groups];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    shuffled.forEach((idxs, gIdx) => {
      const f = gIdx % kFolds;
      for (const idx of idxs) foldAssign[idx] = f;
    });
  }

  return foldAssign;
}

/**
 * Fits Level-2 MAP Ridge Logistic Meta-Learner strictly on train.jsonl OOF features:
 * X = [logit(p_word_oof, 4), logit(p_char_oof, 4), logit(p_ling, 3), incongruity_index, lexical_coverage]
 */
function trainMetaLearner(
  features: number[][],
  labels: number[],
  muPrior: number[],
  lambda = 0.25,
  lr = 0.03,
  epochs = 500
): { bias: number; w: number[]; trainLoss: number } {
  const N = features.length;
  const D = 5;
  let bias = 0.0;
  const w = [...muPrior];

  let mB = 0, vB = 0;
  const mW = new Array(D).fill(0);
  const vW = new Array(D).fill(0);
  const gradW = new Array(D).fill(0);
  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
  const invN = 1.0 / N;

  for (let ep = 1; ep <= epochs; ep++) {
    let gradB = 0.0;
    gradW.fill(0);

    for (let i = 0; i < N; i++) {
      const x = features[i];
      const z = bias + w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3] + w[4] * x[4];
      const p = stableSigmoid(z);
      const err = (p - labels[i]) * invN;
      gradB += err;
      for (let d = 0; d < D; d++) gradW[d] += err * x[d];
    }

    const bc1 = 1.0 - Math.pow(beta1, ep);
    const bc2 = 1.0 - Math.pow(beta2, ep);

    mB = beta1 * mB + (1.0 - beta1) * gradB;
    vB = beta2 * vB + (1.0 - beta2) * gradB * gradB;
    bias -= lr * (mB / bc1) / (Math.sqrt(vB / bc2) + eps);

    for (let d = 0; d < D; d++) {
      const g = gradW[d] + lambda * (w[d] - muPrior[d]);
      mW[d] = beta1 * mW[d] + (1.0 - beta1) * g;
      vW[d] = beta2 * vW[d] + (1.0 - beta2) * g * g;
      w[d] -= lr * (mW[d] / bc1) / (Math.sqrt(vW[d] / bc2) + eps);
    }
  }

  let totalBce = 0;
  for (let i = 0; i < N; i++) {
    const x = features[i];
    const z = bias + w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3] + w[4] * x[4];
    const p = Math.max(1e-12, Math.min(1 - 1e-12, stableSigmoid(z)));
    totalBce -= labels[i] * Math.log(p) + (1 - labels[i]) * Math.log(1 - p);
  }

  return {
    bias: round5(bias),
    w: w.map(round5),
    trainLoss: round4(totalBce / N),
  };
}

/**
 * Fits scalar calibration temperature T* > 0 on val.jsonl minimizing NLL.
 */
function optimizeTemperature(logits: number[], labels: number[]): { temperature: number; nll: number } {
  const evalNll = (T: number) => {
    let nll = 0;
    for (let i = 0; i < logits.length; i++) {
      const p = Math.max(1e-12, Math.min(1 - 1e-12, stableSigmoid(logits[i] / T)));
      nll -= labels[i] * Math.log(p) + (1 - labels[i]) * Math.log(1 - p);
    }
    return nll / logits.length;
  };

  let bestT = 1.0;
  let bestNll = evalNll(1.0);
  for (let T = 0.25; T <= 2.501; T += 0.025) {
    const nll = evalNll(T);
    if (nll < bestNll) {
      bestNll = nll;
      bestT = T;
    }
  }
  // Fine search +/- 0.025 step 0.005
  for (let T = Math.max(0.2, bestT - 0.025); T <= bestT + 0.025; T += 0.005) {
    const nll = evalNll(T);
    if (nll < bestNll) {
      bestNll = nll;
      bestT = T;
    }
  }
  return { temperature: Math.round(bestT * 1000) / 1000, nll: round4(bestNll) };
}

function evaluateBinary(probs: number[], labels: number[]) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  const n = labels.length;
  for (let i = 0; i < n; i++) {
    const pred = probs[i] >= 0.5 ? 1 : 0;
    const y = labels[i];
    if (pred === 1 && y === 1) tp++;
    else if (pred === 1 && y === 0) fp++;
    else if (pred === 0 && y === 0) tn++;
    else fn++;
  }
  const acc = ((tp + tn) / n) * 100;
  const prec = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
  const rec = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
  const f1Sarc = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
  const spec = tn + fp > 0 ? (tn / (tn + fp)) * 100 : 0;
  const precLit = tn + fn > 0 ? (tn / (tn + fn)) * 100 : 0;
  const f1Lit = precLit + spec > 0 ? (2 * precLit * spec) / (precLit + spec) : 0;
  const macroF1 = (f1Sarc + f1Lit) / 2;

  // Guo et al. 5-bin confidence ECE + Brier + NLL
  let ece = 0;
  for (let b = 0; b < 5; b++) {
    const lo = 0.5 + b * 0.1;
    const hi = lo + 0.1;
    let cnt = 0, sConf = 0, sCorr = 0;
    for (let i = 0; i < n; i++) {
      const p = probs[i];
      const conf = Math.max(p, 1 - p);
      const inBin = b === 4 ? conf >= lo && conf <= hi : conf >= lo && conf < hi;
      if (inBin) {
        cnt++;
        sConf += conf;
        sCorr += (p >= 0.5 ? 1 : 0) === labels[i] ? 1 : 0;
      }
    }
    if (cnt > 0) ece += (cnt / n) * Math.abs(sCorr / cnt - sConf / cnt);
  }
  let brier = 0, nll = 0;
  for (let i = 0; i < n; i++) {
    const p = Math.max(1e-12, Math.min(1 - 1e-12, probs[i]));
    brier += (p - labels[i]) ** 2;
    nll -= labels[i] * Math.log(p) + (1 - labels[i]) * Math.log(1 - p);
  }

  return {
    accuracy: round2(acc),
    precision: round2(prec),
    recall: round2(rec),
    macroF1: round2(macroF1),
    specificity: round2(spec),
    ece: round2(ece * 100),
    brier: round4(brier / n),
    nll: round4(nll / n),
    tp, fp, tn, fn,
  };
}

async function main() {
  console.log("================================================================================");
  console.log("SARcasmScope — 3-Stage Calibrated Feature-Fusion Training (train + val ONLY)");
  console.log("================================================================================\n");

  const indepDir = path.resolve("./eval_results/independent");
  const trainRecords = loadJsonl(path.join(indepDir, "train.jsonl"));
  const valRecords = loadJsonl(path.join(indepDir, "val.jsonl"));
  console.log(`Loaded train.jsonl (${trainRecords.length}) & val.jsonl (${valRecords.length}). [test.jsonl UNTOUCHED]`);

  const wordCfg = { minDf: 2, maxFeatures: 8000, l2: 1e-4, lr: 0.05, epochs: 70 };
  const charCfg = { minDf: 3, maxFeatures: 8000, l2: 2e-4, lr: 0.05, epochs: 70 };
  const extractWord = (r: DataRecord) => extractWordFeatures(r.text, r.context);
  const extractChar = (r: DataRecord) => extractCharFeatures(r.text, r.context);

  // 1. 5-Fold Group-Stratified OOF predictions on train.jsonl
  console.log("\n[Stage 1] Running 5-Fold Group-Stratified CV inside train.jsonl...");
  const folds = assignGroupStratifiedFolds(trainRecords, 5, 42);
  const oofWord = new Float64Array(trainRecords.length);
  const oofChar = new Float64Array(trainRecords.length);

  for (let f = 0; f < 5; f++) {
    const trFold: DataRecord[] = [];
    const vaIndices: number[] = [];
    for (let i = 0; i < trainRecords.length; i++) {
      if (folds[i] === f) vaIndices.push(i);
      else trFold.push(trainRecords[i]);
    }
    const wordF = trainSparseLogisticModel(trFold, extractWord, wordCfg);
    const charF = trainSparseLogisticModel(trFold, extractChar, charCfg);
    for (const idx of vaIndices) {
      oofWord[idx] = wordF.predictRecord(trainRecords[idx]);
      oofChar[idx] = charF.predictRecord(trainRecords[idx]);
    }
    console.log(`  Fold ${f + 1}/5 OOF complete (${vaIndices.length} samples held out)`);
  }

  // Extract deterministic Linguistic features on train.jsonl
  const trainLing = trainRecords.map((r) => predictLinguistic(r.text, r.context));
  const trainMetaX = trainRecords.map((_, i) => [
    boundedLogit(oofWord[i], 4.0),
    boundedLogit(oofChar[i], 4.0),
    boundedLogit(trainLing[i].p_sarcastic, 3.0),
    trainLing[i].incongruity.incongruity_index,
    trainLing[i].incongruity.lexical_coverage,
  ]);
  const trainY = trainRecords.map((r) => r.binary_sarcasm_label);

  // 2. Fit Level-2 MAP Ridge Meta-Learner on train.jsonl OOF
  console.log("\n[Stage 2] Fitting MAP Ridge Logistic Meta-Learner strictly on train OOF features...");
  const priorMu = [0.5, 0.5, 0.4, 0.2, 0.2];
  const meta = trainMetaLearner(trainMetaX, trainY, priorMu, 0.25, 0.03, 500);
  console.log("  Learned Meta-Weights:", { bias: meta.bias, w: meta.w, oofTrainBCE: meta.trainLoss });

  // 3. Fit full Level-1 experts on 100% of train.jsonl
  console.log("\n[Stage 2b] Fitting final WordLex and CharSub Level-1 models on full train.jsonl...");
  const finalWord = trainSparseLogisticModel(trainRecords, extractWord, wordCfg);
  const finalChar = trainSparseLogisticModel(trainRecords, extractChar, charCfg);

  // 4. Calibrate Temperature & Selective Abstention threshold strictly on val.jsonl
  console.log("\n[Stage 3] Optimizing Temperature T* and Selective Abstention on val.jsonl (N=1,104)...");
  const valY = valRecords.map((r) => r.binary_sarcasm_label);
  const valLing = valRecords.map((r) => predictLinguistic(r.text, r.context));
  const valWordProbs = valRecords.map((r) => finalWord.predictRecord(r));
  const valCharProbs = valRecords.map((r) => finalChar.predictRecord(r));
  const valStatEnsembleProbs = valRecords.map((_, i) =>
    stableSigmoid((boundedLogit(valWordProbs[i], 4.0) + boundedLogit(valCharProbs[i], 4.0)) * 0.5)
  );
  const valMetaLogits = valRecords.map((_, i) => {
    const x = [
      boundedLogit(valWordProbs[i], 4.0),
      boundedLogit(valCharProbs[i], 4.0),
      boundedLogit(valLing[i].p_sarcastic, 3.0),
      valLing[i].incongruity.incongruity_index,
      valLing[i].incongruity.lexical_coverage,
    ];
    return meta.bias + meta.w[0] * x[0] + meta.w[1] * x[1] + meta.w[2] * x[2] + meta.w[3] * x[3] + meta.w[4] * x[4];
  });

  const calOpt = optimizeTemperature(valMetaLogits, valY);
  const valUncalProbs = valMetaLogits.map((z) => stableSigmoid(z));
  const valCalProbs = valMetaLogits.map((z) => stableSigmoid(z / calOpt.temperature));

  const mValLing = evaluateBinary(valLing.map((l) => l.p_sarcastic), valY);
  const mValWord = evaluateBinary(valWordProbs, valY);
  const mValChar = evaluateBinary(valCharProbs, valY);
  const mValStat = evaluateBinary(valStatEnsembleProbs, valY);
  const mValUncal = evaluateBinary(valUncalProbs, valY);
  const mValCal = evaluateBinary(valCalProbs, valY);

  console.log(`  Optimal Validation Temperature T* = ${calOpt.temperature} (Val NLL: ${mValUncal.nll} -> ${mValCal.nll})`);
  console.log(`  Val Linguistic-Only: Acc=${mValLing.accuracy}% | MF1=${mValLing.macroF1}% | ECE=${mValLing.ece}%`);
  console.log(`  Val WordLex (1-2):   Acc=${mValWord.accuracy}% | MF1=${mValWord.macroF1}% | ECE=${mValWord.ece}%`);
  console.log(`  Val CharSub (3-5):   Acc=${mValChar.accuracy}% | MF1=${mValChar.macroF1}% | ECE=${mValChar.ece}%`);
  console.log(`  Val Uncal Fusion:    Acc=${mValUncal.accuracy}% | MF1=${mValUncal.macroF1}% | ECE=${mValUncal.ece}%`);
  console.log(`  Val Calibrated Fus.: Acc=${mValCal.accuracy}% | MF1=${mValCal.macroF1}% | ECE=${mValCal.ece}%`);

  // Profile Risk-Coverage on val.jsonl using Confidence = 2 * |P - 0.5|
  const taus = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6];
  const valRiskCurve = taus.map((tau) => {
    const accIdx: number[] = [];
    for (let i = 0; i < valCalProbs.length; i++) {
      const conf = Math.abs(valCalProbs[i] - 0.5) * 2.0;
      if (conf >= tau) accIdx.push(i);
    }
    const cov = round2((accIdx.length / valCalProbs.length) * 100);
    if (accIdx.length === 0) return { tau, cov: 0, acc: 0, f1: 0, n: 0 };
    const subP = accIdx.map((i) => valCalProbs[i]);
    const subY = accIdx.map((i) => valY[i]);
    const m = evaluateBinary(subP, subY);
    return { tau, cov, acc: m.accuracy, f1: m.macroF1, n: accIdx.length };
  });

  // Choose smallest tau reaching >= 80% validation selective accuracy with strong coverage
  const chosenPoint = valRiskCurve.find((p) => p.acc >= 80.0 && p.cov >= 30.0) ?? valRiskCurve.find((p) => p.tau === 0.35)!;
  const defaultTau = chosenPoint.tau;
  console.log("\n  Validation Risk-Coverage Curve:");
  for (const r of valRiskCurve) {
    console.log(`    tau=${r.tau.toFixed(2)} | Cov=${String(r.cov).padStart(6)}% (${r.n}/${valY.length}) | SelAcc=${r.acc}% | SelMF1=${r.f1}%`);
  }
  console.log(`  Selected Validation Default Abstention Threshold tau* = ${defaultTau} (Val SelAcc=${chosenPoint.acc}%, Cov=${chosenPoint.cov}%)`);

  // 5. Serialize src/engine/trainedModelData.ts
  const artifact = {
    version: "2.0.0-oof-calibrated",
    trainedOn: "eval_results/independent/train.jsonl (N=3,373)",
    calibratedOn: "eval_results/independent/val.jsonl (N=1,104)",
    seed: 42,
    fusionWeights: {
      w0_bias: meta.bias,
      w1_word: meta.w[0],
      w2_char: meta.w[1],
      w3_linguistic: meta.w[2],
      w4_incongruity: meta.w[3],
      w5_coverage: meta.w[4],
      temperature: calOpt.temperature,
    },
    defaultAbstainThreshold: defaultTau,
    wordModel: finalWord.model,
    charModel: finalChar.model,
  };

  const tsContent = `// AUTO-GENERATED BY eval/train_and_calibrate.ts — DO NOT EDIT MANUALLY
// Fitted strictly on eval_results/independent/train.jsonl (N=3,373) via 5-fold group-stratified OOF stacking
// Post-hoc temperature T* and abstention threshold tau* calibrated strictly on eval_results/independent/val.jsonl (N=1,104)

export interface SerializedSparseModel {
  vocab: string[];
  idf: number[];
  weights: number[];
  bias: number;
}

export interface TrainedModelArtifact {
  version: string;
  trainedOn: string;
  calibratedOn: string;
  seed: number;
  fusionWeights: {
    w0_bias: number;
    w1_word: number;
    w2_char: number;
    w3_linguistic: number;
    w4_incongruity: number;
    w5_coverage: number;
    temperature: number;
  };
  defaultAbstainThreshold: number;
  wordModel: SerializedSparseModel;
  charModel: SerializedSparseModel;
}

export const TRAINED_MODEL_ARTIFACT: TrainedModelArtifact = ${JSON.stringify(artifact)};
`;

  const tsOutPath = path.resolve("./src/engine/trainedModelData.ts");
  fs.writeFileSync(tsOutPath, tsContent);
  const sizeKB = Math.round((Buffer.byteLength(tsContent, "utf8") / 1024) * 10) / 10;
  console.log(`\nWrote serialized model artifact to src/engine/trainedModelData.ts (${sizeKB} KB)`);

  // 6. Write validation summaries & CSVs (without touching test.jsonl)
  const summaryJson = {
    timestamp: new Date().toISOString(),
    splitIntegrity: "Fitted on train.jsonl (N=3,373), calibrated on val.jsonl (N=1,104). test.jsonl untouched.",
    hyperparameters: { wordCfg, charCfg, priorMu, mapLambda: 0.25 },
    fusionWeights: artifact.fusionWeights,
    defaultAbstainThreshold: defaultTau,
    validationMetrics: {
      linguisticOnly: mValLing,
      wordLex: mValWord,
      charSub: mValChar,
      statisticalEnsemble: mValStat,
      uncalibratedHybrid: mValUncal,
      calibratedHybrid: mValCal,
    },
    validationRiskCoverage: valRiskCurve,
  };
  fs.writeFileSync(
    path.join(indepDir, "training_calibration_summary.json"),
    JSON.stringify(summaryJson, null, 2) + "\n"
  );

  const stage3Csv = [
    "System,Accuracy,Precision,Recall,MacroF1,Specificity,ECE,BrierScore,NLL",
    `"Linguistic_Only",${mValLing.accuracy},${mValLing.precision},${mValLing.recall},${mValLing.macroF1},${mValLing.specificity},${mValLing.ece},${mValLing.brier},${mValLing.nll}`,
    `"WordLex_1_2_LR",${mValWord.accuracy},${mValWord.precision},${mValWord.recall},${mValWord.macroF1},${mValWord.specificity},${mValWord.ece},${mValWord.brier},${mValWord.nll}`,
    `"CharSub_3_5_LR",${mValChar.accuracy},${mValChar.precision},${mValChar.recall},${mValChar.macroF1},${mValChar.specificity},${mValChar.ece},${mValChar.brier},${mValChar.nll}`,
    `"Statistical_OOF_Ensemble",${mValStat.accuracy},${mValStat.precision},${mValStat.recall},${mValStat.macroF1},${mValStat.specificity},${mValStat.ece},${mValStat.brier},${mValStat.nll}`,
    `"Uncalibrated_Hybrid_T1",${mValUncal.accuracy},${mValUncal.precision},${mValUncal.recall},${mValUncal.macroF1},${mValUncal.specificity},${mValUncal.ece},${mValUncal.brier},${mValUncal.nll}`,
    `"Calibrated_Hybrid_Fusion",${mValCal.accuracy},${mValCal.precision},${mValCal.recall},${mValCal.macroF1},${mValCal.specificity},${mValCal.ece},${mValCal.brier},${mValCal.nll}`,
  ].join("\n") + "\n";
  fs.writeFileSync(path.join(indepDir, "stage3_system_comparison.csv"), stage3Csv);

  const stage3bCsv = [
    "Representation,VocabSize,Accuracy,Precision,Recall,MacroF1,Specificity,ECE,BrierScore,NLL",
    `"WordLex_1_2_Morph_Phonetic_Context",${finalWord.model.vocab.length},${mValWord.accuracy},${mValWord.precision},${mValWord.recall},${mValWord.macroF1},${mValWord.specificity},${mValWord.ece},${mValWord.brier},${mValWord.nll}`,
    `"CharSub_3_5_Sublexical",${finalChar.model.vocab.length},${mValChar.accuracy},${mValChar.precision},${mValChar.recall},${mValChar.macroF1},${mValChar.specificity},${mValChar.ece},${mValChar.brier},${mValChar.nll}`,
    `"Dual_WordLex_CharSub_Stack",${finalWord.model.vocab.length + finalChar.model.vocab.length},${mValStat.accuracy},${mValStat.precision},${mValStat.recall},${mValStat.macroF1},${mValStat.specificity},${mValStat.ece},${mValStat.brier},${mValStat.nll}`,
    `"Stage2_Symbolic_Linguistic",0,${mValLing.accuracy},${mValLing.precision},${mValLing.recall},${mValLing.macroF1},${mValLing.specificity},${mValLing.ece},${mValLing.brier},${mValLing.nll}`,
    `"Calibrated_3Stage_Feature_Fusion",${finalWord.model.vocab.length + finalChar.model.vocab.length},${mValCal.accuracy},${mValCal.precision},${mValCal.recall},${mValCal.macroF1},${mValCal.specificity},${mValCal.ece},${mValCal.brier},${mValCal.nll}`,
  ].join("\n") + "\n";
  fs.writeFileSync(path.join(indepDir, "stage3b_representations_comparison.csv"), stage3bCsv);
  fs.writeFileSync(path.join(indepDir, "stage3_evaluation_report.json"), JSON.stringify(summaryJson, null, 2) + "\n");

  console.log("Validation calibration artifacts written successfully.\n");
}

main().catch((err) => {
  console.error("Training/calibration failed:", err);
  process.exit(1);
});
