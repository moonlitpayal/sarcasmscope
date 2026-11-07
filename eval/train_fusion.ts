/**
 * SARcasmScope — Stage 3C & 3D: Hybrid Feature Fusion Training & Selective Evaluation
 * 
 * Objectives:
 * 1. Learn regularized probability fusion weights strictly on train.jsonl (3,373 records)
 * 2. Optimize calibration temperature and selective threshold on val.jsonl (1,104 records)
 * 3. Evaluate and compare 3 paradigms:
 *    - System A: Stage 2 Client-Side Linguistic Engine
 *    - System B: Contextual / Subword Statistical Representation Engine
 *    - System C: Calibrated Hybrid Feature-Fusion Engine (SARcasmScope-Hybrid)
 * 4. Measure:
 *    - Accuracy, Precision, Recall, Macro-F1, Weighted-F1, Specificity
 *    - Expected Calibration Error (ECE), Brier Score, NLL
 *    - Selective Prediction Metrics: Coverage, Risk, Selective Accuracy, Abstention Rate
 *    - Language-wise & Dataset-wise breakdowns
 * 
 * [STRICT DATA INTEGRITY: Held-out test.jsonl remains STRICTLY UNTOUCHED]
 */

import * as fs from "fs";
import * as path from "path";
import { predict as predictLinguistic } from "../src/engine/sarcasmEngine";
import {
  predictHybrid,
  calculateECE,
  logit,
  sigmoid,
  DEFAULT_FUSION_WEIGHTS,
  FusionWeights,
} from "../src/engine/hybridFusion";

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

// Tokenize words
function getWords(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((w) => w.length > 0);
}

// Extract subword n-grams for semantic representation
function getSubwords(tokens: string[]): string[] {
  const sw: string[] = [];
  for (const t of tokens) {
    if (t.length <= 4) sw.push(t);
    else {
      for (let i = 0; i <= t.length - 3; i++) {
        sw.push(t.substring(i, i + 3));
      }
    }
  }
  return sw;
}

class FastSemanticModel {
  vocab: Map<string, number> = new Map();
  idf: Float64Array = new Float64Array(0);
  weights: Float64Array = new Float64Array(0);
  bias = 0.0;

  fit(records: DataRecord[]) {
    const docFreq: Map<string, number> = new Map();
    const tokenized = records.map((r) => getSubwords(getWords(`${r.context} ${r.text}`)));

    for (const tokens of tokenized) {
      const seen = new Set(tokens);
      for (const t of seen) {
        docFreq.set(t, (docFreq.get(t) || 0) + 1);
      }
    }

    const candidates = Array.from(docFreq.entries())
      .filter(([_, df]) => df >= 3)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 7500);

    candidates.forEach(([term], idx) => {
      this.vocab.set(term, idx);
    });

    const N = records.length;
    const V = this.vocab.size;
    this.idf = new Float64Array(V);
    this.weights = new Float64Array(V);
    this.bias = 0.0;

    candidates.forEach(([term, df], idx) => {
      this.idf[idx] = Math.log((N + 1) / (df + 1)) + 1.0;
    });

    const lr = 0.04;
    const lambda = 1e-4;
    for (let ep = 0; ep < 20; ep++) {
      for (let i = 0; i < records.length; i++) {
        const y = records[i].binary_sarcasm_label;
        const tokens = tokenized[i];
        const tfMap: Map<number, number> = new Map();
        for (const t of tokens) {
          const idx = this.vocab.get(t);
          if (idx !== undefined) tfMap.set(idx, (tfMap.get(idx) || 0) + 1);
        }

        let dot = this.bias;
        for (const [idx, tf] of tfMap.entries()) {
          dot += this.weights[idx] * (tf * this.idf[idx]);
        }
        const p = 1.0 / (1.0 + Math.exp(-Math.max(-15, Math.min(15, dot))));
        const grad = p - y;

        for (const [idx, tf] of tfMap.entries()) {
          const val = tf * this.idf[idx];
          this.weights[idx] -= lr * (grad * val + lambda * this.weights[idx]);
        }
        this.bias -= lr * grad;
      }
    }
  }

  predict(text: string, context = ""): number {
    const tokens = getSubwords(getWords(`${context} ${text}`));
    const tfMap: Map<number, number> = new Map();
    for (const t of tokens) {
      const idx = this.vocab.get(t);
      if (idx !== undefined) tfMap.set(idx, (tfMap.get(idx) || 0) + 1);
    }
    let dot = this.bias;
    for (const [idx, tf] of tfMap.entries()) {
      dot += this.weights[idx] * (tf * this.idf[idx]);
    }
    return 1.0 / (1.0 + Math.exp(-Math.max(-15, Math.min(15, dot))));
  }
}

function calculateClassificationMetrics(probs: number[], labels: number[], threshold = 0.5) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < probs.length; i++) {
    const pred = probs[i] >= threshold ? 1 : 0;
    const y = labels[i];
    if (pred === 1 && y === 1) tp++;
    else if (pred === 1 && y === 0) fp++;
    else if (pred === 0 && y === 0) tn++;
    else if (pred === 0 && y === 1) fn++;
  }

  const total = labels.length;
  const acc = ((tp + tn) / total) * 100;
  const prec = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
  const rec = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
  const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
  const spec = tn + fp > 0 ? (tn / (tn + fp)) * 100 : 0;

  const precLit = tn + fn > 0 ? (tn / (tn + fn)) * 100 : 0;
  const recLit = tn + fp > 0 ? (tn / (tn + fp)) * 100 : 0;
  const f1Lit = precLit + recLit > 0 ? (2 * precLit * recLit) / (precLit + recLit) : 0;
  const macroF1 = (f1 + f1Lit) / 2;
  const weightedF1 = ((tp + fn) * f1 + (tn + fp) * f1Lit) / total;

  const cal = calculateECE(probs, labels);

  return {
    accuracy: Math.round(acc * 100) / 100,
    precision: Math.round(prec * 100) / 100,
    recall: Math.round(rec * 100) / 100,
    f1: Math.round(f1 * 100) / 100,
    macroF1: Math.round(macroF1 * 100) / 100,
    weightedF1: Math.round(weightedF1 * 100) / 100,
    specificity: Math.round(spec * 100) / 100,
    ece: cal.ece,
    brierScore: cal.brier_score,
    nll: cal.nll,
    tp, fp, tn, fn, total
  };
}

function evaluateSelectivePrediction(
  results: { prob: number; confidence: number; abstain: boolean }[],
  labels: number[]
) {
  const total = results.length;
  const acceptedIndices = results.map((r, i) => (!r.abstain ? i : -1)).filter((i) => i >= 0);
  const acceptedCount = acceptedIndices.length;
  const abstainCount = total - acceptedCount;
  const coverageRate = (acceptedCount / total) * 100;

  if (acceptedCount === 0) {
    return {
      accepted_count: 0,
      abstain_count: total,
      coverage_rate: 0,
      selective_accuracy: 0,
      selective_precision: 0,
      selective_recall: 0,
      selective_macro_f1: 0,
      selective_risk: 0,
    };
  }

  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const idx of acceptedIndices) {
    const pred = results[idx].prob >= 0.5 ? 1 : 0;
    const y = labels[idx];
    if (pred === 1 && y === 1) tp++;
    else if (pred === 1 && y === 0) fp++;
    else if (pred === 0 && y === 0) tn++;
    else if (pred === 0 && y === 1) fn++;
  }

  const selAcc = ((tp + tn) / acceptedCount) * 100;
  const selPrec = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
  const selRec = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
  const selF1 = selPrec + selRec > 0 ? (2 * selPrec * selRec) / (selPrec + selRec) : 0;

  const precLit = tn + fn > 0 ? (tn / (tn + fn)) * 100 : 0;
  const recLit = tn + fp > 0 ? (tn / (tn + fp)) * 100 : 0;
  const f1Lit = precLit + recLit > 0 ? (2 * precLit * recLit) / (precLit + recLit) : 0;
  const selMacroF1 = (selF1 + f1Lit) / 2;
  const selectiveRisk = 100.0 - selAcc;

  return {
    accepted_count: acceptedCount,
    abstain_count: abstainCount,
    coverage_rate: Math.round(coverageRate * 100) / 100,
    selective_accuracy: Math.round(selAcc * 100) / 100,
    selective_precision: Math.round(selPrec * 100) / 100,
    selective_recall: Math.round(selRec * 100) / 100,
    selective_macro_f1: Math.round(selMacroF1 * 100) / 100,
    selective_risk: Math.round(selectiveRisk * 100) / 100,
  };
}

async function runStage3Evaluation() {
  console.log("================================================================================");
  console.log("SARcasmScope — Stage 3C & 3D: Hybrid Fusion Training & Comprehensive Evaluation");
  console.log("================================================================================\n");

  const trainRecords = loadJsonl(path.resolve("./eval_results/independent/train.jsonl"));
  const valRecords = loadJsonl(path.resolve("./eval_results/independent/val.jsonl"));

  console.log(`Training records: ${trainRecords.length} | Validation records: ${valRecords.length}`);

  // 1. Train Subword Contextual Semantic Model on train.jsonl
  console.log("\n[1/4] Training Subword Contextual Model on train.jsonl (3,373 records)...");
  const semModel = new FastSemanticModel();
  semModel.fit(trainRecords);

  // 2. Fit Hybrid Fusion Weights on train.jsonl
  console.log("[2/4] Fitting Regularized Hybrid Fusion Weights on train.jsonl...");
  // Precompute features once
  const trainFeatures = trainRecords.map((r) => {
    const lingRes = predictLinguistic(r.text, r.context);
    const semP = semModel.predict(r.text, r.context);
    return {
      lingLogit: logit(lingRes.p_sarcastic),
      semLogit: logit(semP),
      incIdx: lingRes.incongruity.incongruity_index,
      cov: lingRes.incongruity.lexical_coverage,
      y: r.binary_sarcasm_label,
    };
  });

  // Feature vector: [logit(p_ling), logit(p_sem), incongruity_idx, coverage]
  let w0 = -0.32, w1 = 1.35, w2 = 1.75, w3 = 0.80, w4 = 0.35;
  const lr = 0.01;
  const reg = 1e-4;

  for (let ep = 0; ep < 25; ep++) {
    for (const f of trainFeatures) {
      const z = w0 + w1 * f.lingLogit + w2 * f.semLogit + w3 * f.incIdx + w4 * f.cov;
      const p = sigmoid(z);
      const grad = p - f.y;

      w0 -= lr * grad;
      w1 -= lr * (grad * f.lingLogit + reg * w1);
      w2 -= lr * (grad * f.semLogit + reg * w2);
      w3 -= lr * (grad * f.incIdx + reg * w3);
      w4 -= lr * (grad * f.cov + reg * w4);
    }
  }

  const learnedWeights: FusionWeights = {
    w0_bias: Math.round(w0 * 1000) / 1000,
    w1_linguistic: Math.round(w1 * 1000) / 1000,
    w2_semantic: Math.round(w2 * 1000) / 1000,
    w3_incongruity: Math.round(w3 * 1000) / 1000,
    w4_coverage: Math.round(w4 * 1000) / 1000,
    temperature: 1.15,
  };

  console.log("Learned Fusion Weights:", learnedWeights);

  // 3. Evaluate Models on Independent Validation Split (val.jsonl)
  console.log("\n[3/4] Evaluating Systems on Validation Split (1,104 records)...");
  const valLabels = valRecords.map((r) => r.binary_sarcasm_label);

  // System A: Stage 2 Linguistic-Only
  const lingProbs = valRecords.map((r) => predictLinguistic(r.text, r.context).p_sarcastic);
  const lingSelective = valRecords.map((r) => {
    const res = predictLinguistic(r.text, r.context);
    return { prob: res.p_sarcastic, confidence: res.confidence, abstain: res.abstain };
  });

  // System B: Contextual Semantic-Only
  const semProbs = valRecords.map((r) => semModel.predict(r.text, r.context));
  const semSelective = semProbs.map((p) => {
    const conf = Math.abs(p - 0.5) * 2;
    return { prob: p, confidence: conf, abstain: conf < 0.25 };
  });

  // System C: Calibrated Hybrid Fusion
  const hybridResults = valRecords.map((r) => {
    const semP = semModel.predict(r.text, r.context);
    return predictHybrid(r.text, r.context, {
      p_transformer: semP,
      embedding_source: "subword_hash",
    }, learnedWeights, 0.58);
  });
  const hybridProbs = hybridResults.map((r) => r.p_fused);
  const hybridSelective = hybridResults.map((r) => ({
    prob: r.p_fused,
    confidence: r.confidence,
    abstain: r.abstain,
  }));

  const metricsA = calculateClassificationMetrics(lingProbs, valLabels);
  const metricsB = calculateClassificationMetrics(semProbs, valLabels);
  const metricsC = calculateClassificationMetrics(hybridProbs, valLabels);

  const selA = evaluateSelectivePrediction(lingSelective, valLabels);
  const selB = evaluateSelectivePrediction(semSelective, valLabels);
  const selC = evaluateSelectivePrediction(hybridSelective, valLabels);

  // 4. Print Comparative Results Table
  console.log("\n" + "=".repeat(110));
  console.log(
    "Evaluation System".padEnd(30) +
    "Acc".padEnd(10) +
    "Prec".padEnd(10) +
    "Rec".padEnd(10) +
    "MacroF1".padEnd(10) +
    "Spec".padEnd(10) +
    "ECE".padEnd(10) +
    "Brier".padEnd(10) +
    "Coverage"
  );
  console.log("-".repeat(110));
  console.log(
    "A. Stage 2 Linguistic-Only".padEnd(30) +
    `${metricsA.accuracy}%`.padEnd(10) +
    `${metricsA.precision}%`.padEnd(10) +
    `${metricsA.recall}%`.padEnd(10) +
    `${metricsA.macroF1}%`.padEnd(10) +
    `${metricsA.specificity}%`.padEnd(10) +
    `${metricsA.ece}%`.padEnd(10) +
    `${metricsA.brierScore}`.padEnd(10) +
    `${selA.coverage_rate}%`
  );
  console.log(
    "B. Semantic Subword-Only".padEnd(30) +
    `${metricsB.accuracy}%`.padEnd(10) +
    `${metricsB.precision}%`.padEnd(10) +
    `${metricsB.recall}%`.padEnd(10) +
    `${metricsB.macroF1}%`.padEnd(10) +
    `${metricsB.specificity}%`.padEnd(10) +
    `${metricsB.ece}%`.padEnd(10) +
    `${metricsB.brierScore}`.padEnd(10) +
    `${selB.coverage_rate}%`
  );
  console.log(
    "C. Calibrated Hybrid Fusion".padEnd(30) +
    `${metricsC.accuracy}%`.padEnd(10) +
    `${metricsC.precision}%`.padEnd(10) +
    `${metricsC.recall}%`.padEnd(10) +
    `${metricsC.macroF1}%`.padEnd(10) +
    `${metricsC.specificity}%`.padEnd(10) +
    `${metricsC.ece}%`.padEnd(10) +
    `${metricsC.brierScore}`.padEnd(10) +
    `${selC.coverage_rate}%`
  );
  console.log("=".repeat(110));

  console.log("\n--- Selective Prediction Analysis on Accepted Predictions ---");
  console.log(`System A (Linguistic): Coverage: ${selA.coverage_rate}% | Sel Acc: ${selA.selective_accuracy}% | Sel Macro-F1: ${selA.selective_macro_f1}% | Risk: ${selA.selective_risk}%`);
  console.log(`System B (Semantic):   Coverage: ${selB.coverage_rate}% | Sel Acc: ${selB.selective_accuracy}% | Sel Macro-F1: ${selB.selective_macro_f1}% | Risk: ${selB.selective_risk}%`);
  console.log(`System C (Hybrid):     Coverage: ${selC.coverage_rate}% | Sel Acc: ${selC.selective_accuracy}% | Sel Macro-F1: ${selC.selective_macro_f1}% | Risk: ${selC.selective_risk}%`);

  // 5. Per-Dataset Breakdown for Hybrid Model
  const perDataset: Record<string, { probs: number[]; labels: number[] }> = {};
  for (let i = 0; i < valRecords.length; i++) {
    const ds = valRecords[i].dataset;
    if (!perDataset[ds]) perDataset[ds] = { probs: [], labels: [] };
    perDataset[ds].probs.push(hybridProbs[i]);
    perDataset[ds].labels.push(valLabels[i]);
  }

  console.log("\n--- Per-Dataset Validation Split Breakdown (System C: Hybrid Fusion) ---");
  for (const [dsName, d] of Object.entries(perDataset)) {
    const m = calculateClassificationMetrics(d.probs, d.labels);
    console.log(`* ${dsName.padEnd(30)} | N=${d.labels.length.toString().padEnd(4)} | Acc=${m.accuracy}% Prec=${m.precision}% Rec=${m.recall}% MacroF1=${m.macroF1}% Spec=${m.specificity}%`);
  }

  // 6. Save Comparison Metrics CSV
  const csvPath = path.resolve("./eval_results/independent/stage3_system_comparison.csv");
  const csvContent = [
    "System,Accuracy,Precision,Recall,MacroF1,Specificity,ECE,BrierScore,NLL,AcceptedCoverage,SelectiveAcc,SelectiveRisk",
    `"A_Linguistic_Engine",${metricsA.accuracy},${metricsA.precision},${metricsA.recall},${metricsA.macroF1},${metricsA.specificity},${metricsA.ece},${metricsA.brierScore},${metricsA.nll},${selA.coverage_rate},${selA.selective_accuracy},${selA.selective_risk}`,
    `"B_Semantic_Subword",${metricsB.accuracy},${metricsB.precision},${metricsB.recall},${metricsB.macroF1},${metricsB.specificity},${metricsB.ece},${metricsB.brierScore},${metricsB.nll},${selB.coverage_rate},${selB.selective_accuracy},${selB.selective_risk}`,
    `"C_Hybrid_Fusion",${metricsC.accuracy},${metricsC.precision},${metricsC.recall},${metricsC.macroF1},${metricsC.specificity},${metricsC.ece},${metricsC.brierScore},${metricsC.nll},${selC.coverage_rate},${selC.selective_accuracy},${selC.selective_risk}`,
  ].join("\n") + "\n";

  fs.writeFileSync(csvPath, csvContent);
  console.log(`\nSaved system comparison CSV to: ${csvPath}`);

  // Save Full JSON Report
  const jsonPath = path.resolve("./eval_results/independent/stage3_evaluation_report.json");
  const reportPayload = {
    timestamp: new Date().toISOString(),
    training_split_size: trainRecords.length,
    validation_split_size: valRecords.length,
    learned_fusion_weights: learnedWeights,
    system_a_linguistic: { metrics: metricsA, selective: selA },
    system_b_semantic: { metrics: metricsB, selective: selB },
    system_c_hybrid: {
      metrics: metricsC,
      selective: selC,
      per_dataset: Object.fromEntries(
        Object.entries(perDataset).map(([k, v]) => [k, calculateClassificationMetrics(v.probs, v.labels)])
      ),
    },
  };
  fs.writeFileSync(jsonPath, JSON.stringify(reportPayload, null, 2));
  console.log(`Saved detailed evaluation JSON to: ${jsonPath}`);
}

runStage3Evaluation();
