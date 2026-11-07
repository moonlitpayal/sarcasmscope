/**
 * SARcasmScope — Stage 3B: Embedding and N-Gram Representation Investigation
 * 
 * Compares:
 * 1. Word N-Grams (Unigram + Bigram TF-IDF)
 * 2. Character N-Grams (3-5 Character Shingles)
 * 3. Subword Hash Representations (Fast subword chunking for Indic / Code-Mixed text)
 * 4. Surface Lexical Incongruity Engine (Stage 2 baseline)
 * 
 * Protocol:
 * - Fit all representation spaces strictly on train.jsonl (3,373 records)
 * - Evaluate on val.jsonl (1,104 records)
 * - [NOTE: Held-out test.jsonl remains STRICTLY UNTOUCHED]
 */

import * as fs from "fs";
import * as path from "path";
import { predict } from "../src/engine/sarcasmEngine";

interface DataRecord {
  id: string;
  dataset: string;
  language: string;
  text: string;
  context: string;
  binary_sarcasm_label: 0 | 1;
}

function loadJsonl(filepath: string): DataRecord[] {
  const content = fs.readFileSync(filepath, "utf8").trim().split("\n");
  return content.map((line) => JSON.parse(line));
}

// Tokenize into words
function wordTokens(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((w) => w.length > 0);
}

// Extract word n-grams (1-2)
function extractWordNgrams(tokens: string[]): string[] {
  const ngrams: string[] = [...tokens];
  for (let i = 0; i < tokens.length - 1; i++) {
    ngrams.push(`${tokens[i]}_${tokens[i + 1]}`);
  }
  return ngrams;
}

// Extract character n-grams (3-5)
function extractCharNgrams(text: string): string[] {
  const clean = text.toLowerCase().replace(/\s+/g, " ").trim();
  const ngrams: string[] = [];
  for (let n = 3; n <= 5; n++) {
    for (let i = 0; i <= clean.length - n; i++) {
      ngrams.push(clean.substring(i, i + n));
    }
  }
  return ngrams;
}

// Subword chunker (morpheme-like sliding chunks of length 3-4)
function extractSubwords(tokens: string[]): string[] {
  const subwords: string[] = [];
  for (const t of tokens) {
    if (t.length <= 4) {
      subwords.push(t);
    } else {
      for (let i = 0; i <= t.length - 3; i++) {
        subwords.push(t.substring(i, i + 3));
        if (i + 4 <= t.length) subwords.push(t.substring(i, i + 4));
      }
    }
  }
  return subwords;
}

// Simple TF-IDF Vectorizer & Logistic/Linear Scorer
class SparseLinearClassifier {
  vocab: Map<string, number> = new Map();
  idf: Float64Array = new Float64Array(0);
  weights: Float64Array = new Float64Array(0);
  bias = 0.0;
  name: string;

  constructor(name: string) {
    this.name = name;
  }

  fit(records: DataRecord[], extractor: (r: DataRecord) => string[], minDf = 3, maxFeatures = 8000, lr = 0.05, epochs = 25) {
    console.log(`[${this.name}] Extracting features from ${records.length} training records...`);
    const docFreq: Map<string, number> = new Map();
    const tokenizedDocs = records.map((r) => extractor(r));

    for (const tokens of tokenizedDocs) {
      const seen = new Set(tokens);
      for (const token of seen) {
        docFreq.set(token, (docFreq.get(token) || 0) + 1);
      }
    }

    // Filter by minDf and sort by frequency
    const candidates = Array.from(docFreq.entries())
      .filter(([_, df]) => df >= minDf)
      .sort((a, b) => b[1] - a[1])
      .slice(0, maxFeatures);

    candidates.forEach(([term], idx) => {
      this.vocab.set(term, idx);
    });

    const N = records.length;
    const vocabSize = this.vocab.size;
    this.idf = new Float64Array(vocabSize);
    this.weights = new Float64Array(vocabSize);
    this.bias = 0.0;

    candidates.forEach(([term, df], idx) => {
      this.idf[idx] = Math.log((N + 1) / (df + 1)) + 1.0;
    });

    console.log(`[${this.name}] Vocabulary size: ${vocabSize} features. Training regularized logistic model...`);

    // Stochastic gradient descent with L2 regularization
    const lambda = 1e-4;
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < records.length; i++) {
        const y = records[i].binary_sarcasm_label;
        const tokens = tokenizedDocs[i];

        // Compute term counts
        const tfMap: Map<number, number> = new Map();
        for (const t of tokens) {
          const idx = this.vocab.get(t);
          if (idx !== undefined) {
            tfMap.set(idx, (tfMap.get(idx) || 0) + 1);
          }
        }

        // Dot product
        let dot = this.bias;
        for (const [idx, tf] of tfMap.entries()) {
          const val = tf * this.idf[idx];
          dot += this.weights[idx] * val;
        }

        // Sigmoid
        const p = 1.0 / (1.0 + Math.exp(-Math.max(-15, Math.min(15, dot))));
        const grad = p - y;

        // Weight updates
        for (const [idx, tf] of tfMap.entries()) {
          const val = tf * this.idf[idx];
          this.weights[idx] -= lr * (grad * val + lambda * this.weights[idx]);
        }
        this.bias -= lr * grad;
      }
    }
  }

  predictProb(tokens: string[]): number {
    const tfMap: Map<number, number> = new Map();
    for (const t of tokens) {
      const idx = this.vocab.get(t);
      if (idx !== undefined) {
        tfMap.set(idx, (tfMap.get(idx) || 0) + 1);
      }
    }

    let dot = this.bias;
    for (const [idx, tf] of tfMap.entries()) {
      const val = tf * this.idf[idx];
      dot += this.weights[idx] * val;
    }

    return 1.0 / (1.0 + Math.exp(-Math.max(-15, Math.min(15, dot))));
  }
}

function evaluateModel(
  name: string,
  valRecords: DataRecord[],
  probFn: (r: DataRecord) => number,
  threshold = 0.5
) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  const probs: number[] = [];

  for (const r of valRecords) {
    const p = probFn(r);
    probs.push(p);
    const pred = p >= threshold ? 1 : 0;
    const y = r.binary_sarcasm_label;
    if (pred === 1 && y === 1) tp++;
    else if (pred === 1 && y === 0) fp++;
    else if (pred === 0 && y === 0) tn++;
    else if (pred === 0 && y === 1) fn++;
  }

  const total = valRecords.length;
  const acc = ((tp + tn) / total) * 100;
  const prec = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
  const rec = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
  const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
  const spec = tn + fp > 0 ? (tn / (tn + fp)) * 100 : 0;

  const precLit = tn + fn > 0 ? (tn / (tn + fn)) * 100 : 0;
  const recLit = tn + fp > 0 ? (tn / (tn + fp)) * 100 : 0;
  const f1Lit = precLit + recLit > 0 ? (2 * precLit * recLit) / (precLit + recLit) : 0;
  const macroF1 = (f1 + f1Lit) / 2;

  // Brier Score
  let brier = 0;
  for (let i = 0; i < total; i++) {
    brier += Math.pow(probs[i] - valRecords[i].binary_sarcasm_label, 2);
  }
  brier /= total;

  return {
    name,
    accuracy: acc,
    precision: prec,
    recall: rec,
    f1,
    macroF1,
    specificity: spec,
    brierScore: brier,
    tp, fp, tn, fn, total
  };
}

async function runInvestigation() {
  console.log("================================================================================");
  console.log("SARcasmScope — Stage 3B: Contextual Embedding & Representation Investigation");
  console.log("================================================================================\n");

  const trainPath = path.resolve("./eval_results/independent/train.jsonl");
  const valPath = path.resolve("./eval_results/independent/val.jsonl");

  const trainRecords = loadJsonl(trainPath);
  const valRecords = loadJsonl(valPath);

  console.log(`Loaded Train: ${trainRecords.length} records | Val: ${valRecords.length} records.`);

  // 1. Word N-Gram Model
  const wordModel = new SparseLinearClassifier("Word N-Grams (1-2 gram)");
  wordModel.fit(
    trainRecords,
    (r) => extractWordNgrams(wordTokens(`${r.context} ${r.text}`)),
    3,
    6000
  );

  // 2. Character N-Gram Model (3-5 gram)
  const charModel = new SparseLinearClassifier("Character N-Grams (3-5 char)");
  charModel.fit(
    trainRecords,
    (r) => extractCharNgrams(`${r.context} ${r.text}`),
    5,
    10000
  );

  // 3. Subword Hash Model
  const subwordModel = new SparseLinearClassifier("Subword Chunks (3-4 subwords)");
  subwordModel.fit(
    trainRecords,
    (r) => extractSubwords(wordTokens(`${r.context} ${r.text}`)),
    3,
    8000
  );

  // 4. Evaluate all representation spaces on Validation Split
  console.log("\n--- Evaluating Representation Models on Independent Validation Split (N=1,104) ---");

  // Stage 2 Linguistic Engine
  const lingEval = evaluateModel("Stage 2 Linguistic Engine", valRecords, (r) => {
    return predict(r.text, r.context).p_sarcastic;
  });

  // Word N-Grams
  const wordEval = evaluateModel("Word N-Grams (1-2 gram)", valRecords, (r) => {
    return wordModel.predictProb(extractWordNgrams(wordTokens(`${r.context} ${r.text}`)));
  });

  // Char N-Grams
  const charEval = evaluateModel("Character N-Grams (3-5 char)", valRecords, (r) => {
    return charModel.predictProb(extractCharNgrams(`${r.context} ${r.text}`));
  });

  // Subword Chunks
  const subwordEval = evaluateModel("Subword Chunks (3-4 subwords)", valRecords, (r) => {
    return subwordModel.predictProb(extractSubwords(wordTokens(`${r.context} ${r.text}`)));
  });

  const results = [lingEval, wordEval, charEval, subwordEval];

  console.log("\n" + "=".repeat(105));
  console.log(
    "Representation Model".padEnd(32) +
    "Accuracy".padEnd(12) +
    "Precision".padEnd(12) +
    "Recall".padEnd(12) +
    "Macro-F1".padEnd(12) +
    "Specificity".padEnd(14) +
    "Brier Score"
  );
  console.log("-".repeat(105));

  for (const res of results) {
    console.log(
      res.name.padEnd(32) +
      `${res.accuracy.toFixed(2)}%`.padEnd(12) +
      `${res.precision.toFixed(2)}%`.padEnd(12) +
      `${res.recall.toFixed(2)}%`.padEnd(12) +
      `${res.macroF1.toFixed(2)}%`.padEnd(12) +
      `${res.specificity.toFixed(2)}%`.padEnd(14) +
      res.brierScore.toFixed(4)
    );
  }
  console.log("=".repeat(105));

  // Save results to CSV
  const csvHeader = "Representation,Accuracy,Precision,Recall,F1,MacroF1,Specificity,BrierScore,TP,FP,TN,FN\n";
  const csvRows = results.map((r) =>
    `"${r.name}",${r.accuracy.toFixed(2)},${r.precision.toFixed(2)},${r.recall.toFixed(2)},${r.f1.toFixed(2)},${r.macroF1.toFixed(2)},${r.specificity.toFixed(2)},${r.brierScore.toFixed(4)},${r.tp},${r.fp},${r.tn},${r.fn}`
  );

  const outPath = path.resolve("./eval_results/independent/stage3b_representations_comparison.csv");
  fs.writeFileSync(outPath, csvHeader + csvRows.join("\n") + "\n");
  console.log(`\nSaved representation comparison metrics to: ${outPath}`);
}

runInvestigation();
