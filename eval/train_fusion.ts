/**
 * SARcasmScope — Validation Fusion Diagnostic Runner
 *
 * Evaluates the fitted 3-Stage Calibrated Feature-Fusion artifacts from
 * `src/engine/trainedModelData.ts` (produced by `eval/train_and_calibrate.ts`)
 * across `val.jsonl` without overwriting the canonical training summary.
 */

import * as fs from "fs";
import * as path from "path";
import { predict as predictLinguistic } from "../src/engine/sarcasmEngine";
import { predictStatistical } from "../src/engine/statisticalModel";
import {
  predictHybrid,
  calculateECE,
  DEFAULT_FUSION_WEIGHTS,
  DEFAULT_ABSTAIN_THRESHOLD,
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
  return fs
    .readFileSync(filepath, "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
}

export async function runValidationFusionDiagnostic() {
  const valRecords = loadJsonl(path.resolve("./eval_results/independent/val.jsonl"));
  const valLabels = valRecords.map((r) => r.binary_sarcasm_label);

  const lingProbs = valRecords.map((r) => predictLinguistic(r.text, r.context).p_sarcastic);
  const statProbs = valRecords.map((r) => predictStatistical(r.text, r.context).p_statistical);
  const hybResults = valRecords.map((r) => predictHybrid(r.text, r.context));
  const hybProbs = hybResults.map((r) => r.p_fused);

  const report = (label: string, probs: number[]) => {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < valLabels.length; i++) {
      const p = probs[i] >= 0.5 ? 1 : 0;
      const y = valLabels[i];
      if (p === 1 && y === 1) tp++;
      else if (p === 1 && y === 0) fp++;
      else if (p === 0 && y === 0) tn++;
      else fn++;
    }
    const acc = ((tp + tn) / valLabels.length) * 100;
    const prec = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
    const rec = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
    const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
    const prec0 = tn + fn > 0 ? (tn / (tn + fn)) * 100 : 0;
    const rec0 = tn + fp > 0 ? (tn / (tn + fp)) * 100 : 0;
    const f10 = prec0 + rec0 > 0 ? (2 * prec0 * rec0) / (prec0 + rec0) : 0;
    const mf1 = (f1 + f10) / 2;
    const cal = calculateECE(probs, valLabels);
    console.log(
      `${label.padEnd(32)} Acc=${acc.toFixed(2)}% | MF1=${mf1.toFixed(2)}% | ECE=${cal.ece}% | Brier=${cal.brier_score} | NLL=${cal.nll}`
    );
  };

  console.log("=== STAGE 3 VALIDATION DIAGNOSTIC (val.jsonl, N=" + valRecords.length + ") ===");
  console.log("Learned Fusion Weights:", DEFAULT_FUSION_WEIGHTS);
  console.log(`Default Abstention Threshold tau*: ${DEFAULT_ABSTAIN_THRESHOLD}\n`);
  report("Linguistic-Only (v2)", lingProbs);
  report("Level-1 Statistical Ensemble", statProbs);
  report("Calibrated 3-Stage Hybrid", hybProbs);

  const accepted = hybResults.filter((r) => !r.abstain);
  const accCorrect = hybResults.filter(
    (r, idx) => !r.abstain && (r.p_fused >= 0.5 ? 1 : 0) === valLabels[idx]
  ).length;
  console.log(
    `Selective Coverage @ tau=${DEFAULT_ABSTAIN_THRESHOLD}: ${accepted.length}/${valLabels.length} (${((accepted.length / valLabels.length) * 100).toFixed(2)}%) | Selective Acc: ${accepted.length ? ((accCorrect / accepted.length) * 100).toFixed(2) : 0}%`
  );
}

const isMain =
  typeof process !== "undefined" &&
  process.argv[1] &&
  (process.argv[1].endsWith("train_fusion.ts") ||
    process.argv[1].endsWith("train_fusion.js"));

if (isMain) {
  runValidationFusionDiagnostic().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
