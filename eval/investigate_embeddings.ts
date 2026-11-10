/**
 * SARcasmScope — Stage 3B: Representation & Expert Investigation
 *
 * Compares validation-split performance (val.jsonl, N = 1,104) of:
 *   1. Stage 2 Symbolic Linguistic Engine
 *   2. WordLex (Word 1-2 + Indic stems + Romanized phonetic keys + `cw:` context)
 *   3. CharSub (Character 3-5 boundary-marked sublexical shingles)
 *   4. Dual Statistical OOF Ensemble (WordLex + CharSub)
 *   5. Calibrated 3-Stage Hybrid Feature Fusion (WordLex + CharSub + Linguistic + T*)
 */

import * as fs from "fs";
import * as path from "path";
import { predict as predictLinguistic } from "../src/engine/sarcasmEngine";
import { scoreWordModel, scoreCharModel, predictStatistical } from "../src/engine/statisticalModel";
import { predictHybrid, calculateECE } from "../src/engine/hybridFusion";

interface DataRecord {
  id: string;
  dataset: string;
  language: string;
  text: string;
  context: string;
  binary_sarcasm_label: 0 | 1;
}

function loadJsonl(filepath: string): DataRecord[] {
  return fs
    .readFileSync(filepath, "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
}

function evaluateModel(name: string, valRecords: DataRecord[], probFn: (r: DataRecord) => number) {
  const probs = valRecords.map(probFn);
  const labels = valRecords.map((r) => r.binary_sarcasm_label);
  let tp = 0, fp = 0, tn = 0, fn = 0;

  for (let i = 0; i < labels.length; i++) {
    const pred = probs[i] >= 0.5 ? 1 : 0;
    const y = labels[i];
    if (pred === 1 && y === 1) tp++;
    else if (pred === 1 && y === 0) fp++;
    else if (pred === 0 && y === 0) tn++;
    else fn++;
  }

  const total = labels.length;
  const acc = ((tp + tn) / total) * 100;
  const prec = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
  const rec = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
  const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
  const spec = tn + fp > 0 ? (tn / (tn + fp)) * 100 : 0;
  const precLit = tn + fn > 0 ? (tn / (tn + fn)) * 100 : 0;
  const f1Lit = precLit + spec > 0 ? (2 * precLit * spec) / (precLit + spec) : 0;
  const macroF1 = (f1 + f1Lit) / 2;
  const cal = calculateECE(probs, labels);

  return {
    name,
    accuracy: Math.round(acc * 100) / 100,
    precision: Math.round(prec * 100) / 100,
    recall: Math.round(rec * 100) / 100,
    f1: Math.round(f1 * 100) / 100,
    macroF1: Math.round(macroF1 * 100) / 100,
    specificity: Math.round(spec * 100) / 100,
    ece: cal.ece,
    brierScore: cal.brier_score,
    nll: cal.nll,
    tp, fp, tn, fn, total,
  };
}

export async function runInvestigation() {
  const valRecords = loadJsonl(path.resolve("./eval_results/independent/val.jsonl"));
  const results = [
    evaluateModel("Stage 2 Symbolic Linguistic", valRecords, (r) => predictLinguistic(r.text, r.context).p_sarcastic),
    evaluateModel("WordLex (1-2 + Morph + Phon + Ctx)", valRecords, (r) => scoreWordModel(r.text, r.context).prob),
    evaluateModel("CharSub (3-5 Sublexical)", valRecords, (r) => scoreCharModel(r.text, r.context).prob),
    evaluateModel("Dual Statistical Ensemble", valRecords, (r) => predictStatistical(r.text, r.context).p_statistical),
    evaluateModel("Calibrated 3-Stage Hybrid Fusion", valRecords, (r) => predictHybrid(r.text, r.context).p_fused),
  ];

  console.log("\n=== VALIDATION REPRESENTATION COMPARISON (val.jsonl, N=" + valRecords.length + ") ===");
  for (const r of results) {
    console.log(
      `${r.name.padEnd(36)} Acc=${r.accuracy}% | MF1=${r.macroF1}% | ECE=${r.ece}% | Brier=${r.brierScore}`
    );
  }
  return results;
}

const isMain =
  typeof process !== "undefined" &&
  process.argv[1] &&
  (process.argv[1].endsWith("investigate_embeddings.ts") ||
    process.argv[1].endsWith("investigate_embeddings.js"));

if (isMain) {
  runInvestigation().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
