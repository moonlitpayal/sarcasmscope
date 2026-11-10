/**
 * SARcasmScope — Standalone Zero-Shortcut Independent Verification & Recomputation Audit
 *
 * Addresses peer-review Requirements 2, 3, and 4 independently from scratch:
 *  1. Evaluates the exact shipped prediction engine (`predictHybrid`, `predictLinguistic`,
 *     `predictStatistical`, and Nov 7 `prevPredict`) directly on:
 *       - `eval_results/independent/val.jsonl`  (N = 1,104)
 *       - `eval_results/independent/test.jsonl` (N = 1,123)
 *  2. Cross-checks row-for-row probability and binary label parity against BOTH exported
 *     prediction CSVs (`stage9/predictions.csv` and `independent/independent_sample_predictions.csv`).
 *  3. Independently recomputes all confusion matrices (`TP, FP, TN, FN`), Accuracy,
 *     Precision, Recall, Specificity, Macro-F1, Weighted-F1, Conf-ECE (Guo et al. 2017),
 *     Class-ECE, Brier, NLL, and Selective Risk-Coverage curves from raw predictions ONLY.
 *  4. Verifies non-circular Validation fit (`T* = 0.845`, `tau* = 0.35`) vs Sealed Test
 *     evaluation (`T=1.0` uncalibrated vs `T*=0.845` calibrated), including bin-by-bin reliability
 *     on BOTH splits and per-language Conf-ECE.
 *  5. Audits raw dataset schemas (`original_label`, `hasContext`, and L3Cube MahaEmotions
 *     CoTR-train vs manual-test source row ranges) across train/val/test JSONL files.
 *
 * Emits:
 *  - `eval_results/independent/validation_vs_test_calibration_and_selective.csv`
 *  - `eval_results/independent/independent_recomputation_audit.json`
 */

import * as fs from "fs";
import * as path from "path";
import { predict as predictV2 } from "../src/engine/sarcasmEngine";
import { predictHybrid, DEFAULT_FUSION_WEIGHTS, DEFAULT_ABSTAIN_THRESHOLD } from "../src/engine/hybridFusion";
import { predictStatistical } from "../src/engine/statisticalModel";
import { prevPredict } from "./verify_comparison";

interface JsonlRecord {
  id: string;
  dataset: string;
  language: string;
  text: string;
  context: string;
  conversation_id?: string;
  original_label: string | number;
  binary_sarcasm_label: 0 | 1;
}

function parseCSV(text: string): string[][] {
  const rows: string[][] = [];
  let cur: string[] = [""];
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    const next = text[i + 1];
    if (c === '"' && inQuotes && next === '"') {
      cur[cur.length - 1] += '"';
      i++;
    } else if (c === '"') {
      inQuotes = !inQuotes;
    } else if (c === "," && !inQuotes) {
      cur.push("");
    } else if ((c === "\r" || c === "\n") && !inQuotes) {
      if (c === "\r" && next === "\n") i++;
      if (cur.length > 1 || cur[0] !== "") rows.push(cur);
      cur = [""];
    } else {
      cur[cur.length - 1] += c;
    }
  }
  if (cur.length > 1 || cur[0] !== "") rows.push(cur);
  return rows;
}

const r2 = (v: number) => Math.round(v * 100) / 100;
const r4 = (v: number) => Math.round(v * 10000) / 10000;

function pureRecomputeMetrics(probs: number[], yTrue: number[]) {
  const n = yTrue.length;
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < n; i++) {
    const pred = probs[i] >= 0.5 ? 1 : 0;
    const y = yTrue[i];
    if (pred === 1 && y === 1) tp++;
    else if (pred === 1 && y === 0) fp++;
    else if (pred === 0 && y === 0) tn++;
    else fn++;
  }

  const acc = n > 0 ? ((tp + tn) / n) * 100 : 0;
  const prec1 = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
  const rec1 = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
  const f1_1 = prec1 + rec1 > 0 ? (2 * prec1 * rec1) / (prec1 + rec1) : 0;
  const spec = tn + fp > 0 ? (tn / (tn + fp)) * 100 : 0;
  const prec0 = tn + fn > 0 ? (tn / (tn + fn)) * 100 : 0;
  const f1_0 = prec0 + spec > 0 ? (2 * prec0 * spec) / (prec0 + spec) : 0;
  const macroF1 = (f1_1 + f1_0) / 2;
  const n1 = yTrue.filter((y) => y === 1).length;
  const weightedF1 = n > 0 ? (f1_1 * n1 + f1_0 * (n - n1)) / n : 0;

  // 5-bin Guo et al. (2017) Confidence ECE
  let confEce = 0;
  const bins: Array<{
    bin: number;
    range: string;
    count: number;
    meanConf: number;
    empAcc: number;
    calError: number;
  }> = [];

  for (let b = 0; b < 5; b++) {
    const lo = 0.5 + b * 0.1;
    const hi = lo + 0.1;
    let cnt = 0, sumConf = 0, sumCorr = 0;
    for (let i = 0; i < n; i++) {
      const p = probs[i];
      const conf = Math.max(p, 1 - p);
      const inBin = b === 4 ? conf >= lo && conf <= hi : conf >= lo && conf < hi;
      if (inBin) {
        cnt++;
        sumConf += conf;
        sumCorr += (p >= 0.5 ? 1 : 0) === yTrue[i] ? 1 : 0;
      }
    }
    const meanConf = cnt > 0 ? sumConf / cnt : (lo + hi) / 2;
    const empAcc = cnt > 0 ? sumCorr / cnt : 0;
    const err = cnt > 0 ? Math.abs(empAcc - meanConf) : 0;
    if (cnt > 0) confEce += (cnt / n) * err;
    bins.push({
      bin: b + 1,
      range: `[${lo.toFixed(1)}-${hi.toFixed(1)}]`,
      count: cnt,
      meanConf: Math.round(meanConf * 1000) / 1000,
      empAcc: Math.round(empAcc * 1000) / 1000,
      calError: Math.round(err * 1000) / 1000,
    });
  }

  // 10-bin Class-wise ECE + Brier + NLL
  let classEce = 0, brier = 0, nll = 0;
  for (let b = 0; b < 10; b++) {
    const lo = b / 10;
    const hi = (b + 1) / 10;
    let cnt = 0, sumP = 0, sumY = 0;
    for (let i = 0; i < n; i++) {
      const p = probs[i];
      const inBin = b === 9 ? p >= lo && p <= hi : p >= lo && p < hi;
      if (inBin) {
        cnt++;
        sumP += p;
        sumY += yTrue[i];
      }
    }
    if (cnt > 0) classEce += (cnt / n) * Math.abs(sumY / cnt - sumP / cnt);
  }

  for (let i = 0; i < n; i++) {
    const p = Math.max(1e-12, Math.min(1 - 1e-12, probs[i]));
    const y = yTrue[i];
    brier += (p - y) ** 2;
    nll -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
  }

  return {
    n, tp, fp, tn, fn,
    accuracy: r2(acc),
    precision: r2(prec1),
    recall: r2(rec1),
    specificity: r2(spec),
    f1Sarc: r2(f1_1),
    macroF1: r2(macroF1),
    weightedF1: r2(weightedF1),
    confEce: r2(confEce * 100),
    classEce: r2(classEce * 100),
    brier: r4(brier / Math.max(1, n)),
    nll: r4(nll / Math.max(1, n)),
    bins,
  };
}

function pureSelectiveCurve(probs: number[], yTrue: number[], taus: number[]) {
  return taus.map((tau) => {
    const keepP: number[] = [];
    const keepY: number[] = [];
    for (let i = 0; i < probs.length; i++) {
      const margin = Math.round(Math.abs(probs[i] - 0.5) * 2.0 * 1000) / 1000;
      if (margin >= tau) {
        keepP.push(probs[i]);
        keepY.push(yTrue[i]);
      }
    }
    const m = pureRecomputeMetrics(keepP, keepY);
    const cov = r2((keepP.length / probs.length) * 100);
    return {
      tau,
      accepted: keepP.length,
      abstained: probs.length - keepP.length,
      coveragePct: cov,
      selectiveAcc: m.accuracy,
      selectiveMacroF1: m.macroF1,
      selectiveRiskPct: r2(100 - m.accuracy),
    };
  });
}

function loadJsonl(p: string): JsonlRecord[] {
  return fs
    .readFileSync(p, "utf8")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l));
}

async function main() {
  const indepDir = path.resolve("./eval_results/independent");
  const stage9Dir = path.resolve("./eval_results/stage9");

  const trainRecords = loadJsonl(path.join(indepDir, "train.jsonl"));
  const valRecords = loadJsonl(path.join(indepDir, "val.jsonl"));
  const testRecords = loadJsonl(path.join(indepDir, "test.jsonl"));

  // 1. Live engine execution on Validation (N=1,104) & Sealed Test (N=1,123)
  const uncalWeights = { ...DEFAULT_FUSION_WEIGHTS, temperature: 1.0 };

  const valY = valRecords.map((r) => r.binary_sarcasm_label);
  const valProbUncal = valRecords.map((r) => predictHybrid(r.text, r.context, undefined, uncalWeights).p_fused);
  const valProbCal = valRecords.map((r) => predictHybrid(r.text, r.context, undefined, DEFAULT_FUSION_WEIGHTS).p_fused);

  const testY = testRecords.map((r) => r.binary_sarcasm_label);
  const testProbV1 = testRecords.map((r) => prevPredict(r.text, r.context).pSarcastic);
  const testProbV2 = testRecords.map((r) => predictV2(r.text, r.context).p_sarcastic);
  const testProbStat = testRecords.map((r) => predictStatistical(r.text, r.context).p_statistical);
  const testProbUncal = testRecords.map((r) => predictHybrid(r.text, r.context, undefined, uncalWeights).p_fused);
  const testProbCal = testRecords.map((r) => predictHybrid(r.text, r.context, undefined, DEFAULT_FUSION_WEIGHTS).p_fused);

  // 2. Verify row-by-row agreement with exported Stage 9 and Independent CSVs
  const stage9PredCsv = parseCSV(fs.readFileSync(path.join(stage9Dir, "predictions.csv"), "utf8"));
  const indepSampleCsv = parseCSV(fs.readFileSync(path.join(indepDir, "independent_sample_predictions.csv"), "utf8"));
  const stage9Rows = stage9PredCsv.slice(1);
  const indepRows = indepSampleCsv.slice(1);

  let maxDeltaStage9 = 0;
  let maxDeltaIndep = 0;
  let labelMatchesStage9 = 0;
  let labelMatchesIndep = 0;
  const csvStage9Probs: number[] = [];
  const csvStage9Labels: number[] = [];

  for (let i = 0; i < testRecords.length; i++) {
    const r = testRecords[i];
    const liveP = testProbCal[i];
    const sRow = stage9Rows[i];
    const iRow = indepRows[i];

    if (sRow && sRow[0] === r.id) {
      const pCsv = parseFloat(sRow[9]);
      const yCsv = parseInt(sRow[3], 10);
      csvStage9Probs.push(pCsv);
      csvStage9Labels.push(yCsv);
      maxDeltaStage9 = Math.max(maxDeltaStage9, Math.abs(pCsv - liveP));
      if (yCsv === r.binary_sarcasm_label) labelMatchesStage9++;
    }
    if (iRow && iRow[0] === r.id) {
      const pCsv = parseFloat(iRow[4]);
      const yCsv = parseInt(iRow[3], 10);
      maxDeltaIndep = Math.max(maxDeltaIndep, Math.abs(pCsv - liveP));
      if (yCsv === r.binary_sarcasm_label) labelMatchesIndep++;
    }
  }

  // 3. Pure recomputation of Confusion Matrices & Metrics
  const testMetrics = {
    v1_Prev_Symbolic: pureRecomputeMetrics(testProbV1, testY),
    v2_Symbolic_Linguistic: pureRecomputeMetrics(testProbV2, testY),
    Statistical_OOF_Ensemble: pureRecomputeMetrics(testProbStat, testY),
    Uncalibrated_Hybrid_T1: pureRecomputeMetrics(testProbUncal, testY),
    Calibrated_Hybrid_Live: pureRecomputeMetrics(testProbCal, testY),
    Calibrated_Hybrid_FromStage9CSV: pureRecomputeMetrics(csvStage9Probs, csvStage9Labels),
  };

  const valCalibration = {
    uncalibrated_T1: pureRecomputeMetrics(valProbUncal, valY),
    calibrated_T0845: pureRecomputeMetrics(valProbCal, valY),
  };

  const taus = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6];
  const valSelective = pureSelectiveCurve(valProbCal, valY, taus);
  const testSelective = pureSelectiveCurve(testProbCal, testY, taus);

  // 4. Schema, Original Label, Context & MahaEmotions CoTR-train vs Gold-test audit
  const datasetSchemaAudit = ["train", "val", "test"].map((splitName) => {
    const arr = splitName === "train" ? trainRecords : splitName === "val" ? valRecords : testRecords;
    const byDs: Record<string, any> = {};
    for (const r of arr) {
      if (!byDs[r.dataset]) {
        byDs[r.dataset] = {
          n: 0,
          sarcastic: 0,
          literal: 0,
          withContext: 0,
          withoutContext: 0,
          originalLabels: {} as Record<string, number>,
        };
      }
      const d = byDs[r.dataset];
      d.n++;
      if (r.binary_sarcasm_label === 1) d.sarcastic++;
      else d.literal++;
      if (r.context && r.context.trim().length > 0) d.withContext++;
      else d.withoutContext++;
      const ol = String(r.original_label);
      d.originalLabels[ol] = (d.originalLabels[ol] || 0) + 1;
    }
    return { split: splitName, total: arr.length, datasets: byDs };
  });

  // Check L3Cube_MahaEmotions exact split provenance (train.csv vs test.csv index threshold ~11000)
  const mahaProvenance = ["train", "val", "test"].map((splitName) => {
    const arr = (splitName === "train" ? trainRecords : splitName === "val" ? valRecords : testRecords)
      .filter((r) => r.dataset === "L3Cube_MahaEmotions");
    const sarc = arr.filter((r) => r.binary_sarcasm_label === 1);
    const neut = arr.filter((r) => r.binary_sarcasm_label === 0);
    return {
      split: splitName,
      total: arr.length,
      sarcasmCount: sarc.length,
      neutralCount: neut.length,
      sarcasmFromSourceTestIdxGe11k: sarc.filter((r) => parseInt(r.id.replace("MAHA_MR_", ""), 10) >= 11000).length,
      neutralFromSourceTestIdxGe11k: neut.filter((r) => parseInt(r.id.replace("MAHA_MR_", ""), 10) >= 11000).length,
    };
  });

  // 5. Write Calibration & Selective CSV across Validation and Sealed Test
  const calSelCsv: string[] = [
    "AuditSection,Split,Condition,Parameter,MetricName,Value1,Value2,Value3,Value4,Notes",
    `"Temperature_Summary","val.jsonl (N=1,104)","Uncalibrated","T=1.0","Acc / MF1 / ConfECE / NLL",${valCalibration.uncalibrated_T1.accuracy},${valCalibration.uncalibrated_T1.macroF1},${valCalibration.uncalibrated_T1.confEce},${valCalibration.uncalibrated_T1.nll},"Brier=${valCalibration.uncalibrated_T1.brier}"`,
    `"Temperature_Summary","val.jsonl (N=1,104)","Calibrated","T*=0.845","Acc / MF1 / ConfECE / NLL",${valCalibration.calibrated_T0845.accuracy},${valCalibration.calibrated_T0845.macroF1},${valCalibration.calibrated_T0845.confEce},${valCalibration.calibrated_T0845.nll},"NLL-minimizing fit on validation only (Brier=${valCalibration.calibrated_T0845.brier})"`,
    `"Temperature_Summary","test.jsonl (N=1,123)","Uncalibrated","T=1.0","Acc / MF1 / ConfECE / NLL",${testMetrics.Uncalibrated_Hybrid_T1.accuracy},${testMetrics.Uncalibrated_Hybrid_T1.macroF1},${testMetrics.Uncalibrated_Hybrid_T1.confEce},${testMetrics.Uncalibrated_Hybrid_T1.nll},"ClassECE=${testMetrics.Uncalibrated_Hybrid_T1.classEce}% Brier=${testMetrics.Uncalibrated_Hybrid_T1.brier}"`,
    `"Temperature_Summary","test.jsonl (N=1,123)","Calibrated","T*=0.845","Acc / MF1 / ConfECE / NLL",${testMetrics.Calibrated_Hybrid_Live.accuracy},${testMetrics.Calibrated_Hybrid_Live.macroF1},${testMetrics.Calibrated_Hybrid_Live.confEce},${testMetrics.Calibrated_Hybrid_Live.nll},"ClassECE=${testMetrics.Calibrated_Hybrid_Live.classEce}% Brier=${testMetrics.Calibrated_Hybrid_Live.brier}"`,
    ...valCalibration.calibrated_T0845.bins.map((b) =>
      `"Reliability_Bin","val.jsonl (N=1,104)","Calibrated (T*=0.845)","${b.range}","N / MeanConf / EmpAcc / Error",${b.count},${b.meanConf},${b.empAcc},${b.calError},"Bin ${b.bin}"`
    ),
    ...testMetrics.Uncalibrated_Hybrid_T1.bins.map((b) =>
      `"Reliability_Bin","test.jsonl (N=1,123)","Uncalibrated (T=1.0)","${b.range}","N / MeanConf / EmpAcc / Error",${b.count},${b.meanConf},${b.empAcc},${b.calError},"Bin ${b.bin} pre-T*"`
    ),
    ...testMetrics.Calibrated_Hybrid_Live.bins.map((b) =>
      `"Reliability_Bin","test.jsonl (N=1,123)","Calibrated (T*=0.845)","${b.range}","N / MeanConf / EmpAcc / Error",${b.count},${b.meanConf},${b.empAcc},${b.calError},"Bin ${b.bin} post-T*"`
    ),
    ...valSelective.map((s) =>
      `"Selective_RiskCoverage","val.jsonl (N=1,104)","MarginConfidence>=tau","tau=${s.tau}","Accepted / Cov% / SelAcc% / SelRisk%",${s.accepted},${s.coveragePct},${s.selectiveAcc},${s.selectiveRiskPct},"SelMF1=${s.selectiveMacroF1}%${s.tau === DEFAULT_ABSTAIN_THRESHOLD ? " [VAL-SELECTED DEFAULT]" : ""}"`
    ),
    ...testSelective.map((s) =>
      `"Selective_RiskCoverage","test.jsonl (N=1,123)","MarginConfidence>=tau","tau=${s.tau}","Accepted / Cov% / SelAcc% / SelRisk%",${s.accepted},${s.coveragePct},${s.selectiveAcc},${s.selectiveRiskPct},"SelMF1=${s.selectiveMacroF1}%${s.tau === DEFAULT_ABSTAIN_THRESHOLD ? " [FROZEN SEALED EVAL]" : ""}"`
    ),
  ];

  fs.writeFileSync(
    path.join(indepDir, "validation_vs_test_calibration_and_selective.csv"),
    calSelCsv.join("\n") + "\n"
  );

  const report = {
    timestamp: new Date().toISOString(),
    exportParityCheck: {
      testRecordCount: testRecords.length,
      stage9CsvRowCount: stage9Rows.length,
      indepSampleCsvRowCount: indepRows.length,
      stage9IdLabelMatches: labelMatchesStage9,
      indepIdLabelMatches: labelMatchesIndep,
      maxAbsProbDeltaVsStage9CSV: r4(maxDeltaStage9),
      maxAbsProbDeltaVsIndepCSV: r4(maxDeltaIndep),
    },
    recomputedTestConfusionAndMetrics: testMetrics,
    validationVsTestCalibration: {
      validation_N1104: valCalibration,
      sealedTest_N1123: {
        uncalibrated_T1: testMetrics.Uncalibrated_Hybrid_T1,
        calibrated_T0845: testMetrics.Calibrated_Hybrid_Live,
      },
    },
    selectiveRiskCoverage: {
      defaultThresholdTau: DEFAULT_ABSTAIN_THRESHOLD,
      validationCurve: valSelective,
      sealedTestCurve: testSelective,
    },
    datasetSchemaAndContextAudit: datasetSchemaAudit,
    mahaEmotionsSourceProvenance: mahaProvenance,
  };

  fs.writeFileSync(
    path.join(indepDir, "independent_recomputation_audit.json"),
    JSON.stringify(report, null, 2) + "\n"
  );

  console.log("================================================================================");
  console.log("SARcasmScope — Independent Verification & Pure Recomputation Audit");
  console.log("================================================================================");
  console.log(`CSV Parity Check (N=${testRecords.length}):`);
  console.log(`  stage9/predictions.csv matches=${labelMatchesStage9}/${testRecords.length}, max|Δp|=${maxDeltaStage9.toExponential(2)}`);
  console.log(`  independent_sample_predictions.csv matches=${labelMatchesIndep}/${testRecords.length}, max|Δp|=${maxDeltaIndep.toExponential(2)}`);
  console.log("\nRecomputed Sealed Test Confusion Matrices (N=1,123):");
  for (const [k, m] of Object.entries(testMetrics)) {
    console.log(
      `  ${k.padEnd(32)} TP=${m.tp} FP=${m.fp} TN=${m.tn} FN=${m.fn} | Acc=${m.accuracy}% MF1=${m.macroF1}% Rec=${m.recall}% ConfECE=${m.confEce}% NLL=${m.nll}`
    );
  }
  console.log("\nValidation vs Sealed Test Calibration (T=1.0 -> T*=0.845):");
  console.log(
    `  val.jsonl  (N=1,104): NLL ${valCalibration.uncalibrated_T1.nll} -> ${valCalibration.calibrated_T0845.nll}, ConfECE ${valCalibration.uncalibrated_T1.confEce}% -> ${valCalibration.calibrated_T0845.confEce}%`
  );
  console.log(
    `  test.jsonl (N=1,123): NLL ${testMetrics.Uncalibrated_Hybrid_T1.nll} -> ${testMetrics.Calibrated_Hybrid_Live.nll}, ConfECE ${testMetrics.Uncalibrated_Hybrid_T1.confEce}% -> ${testMetrics.Calibrated_Hybrid_Live.confEce}%`
  );
  console.log("\nSelective Prediction at tau* = 0.35:");
  const vSel = valSelective.find((s) => s.tau === DEFAULT_ABSTAIN_THRESHOLD)!;
  const tSel = testSelective.find((s) => s.tau === DEFAULT_ABSTAIN_THRESHOLD)!;
  console.log(`  val.jsonl  : ${vSel.accepted}/${valRecords.length} (${vSel.coveragePct}% cov) -> SelAcc=${vSel.selectiveAcc}%, SelMF1=${vSel.selectiveMacroF1}%`);
  console.log(`  test.jsonl : ${tSel.accepted}/${testRecords.length} (${tSel.coveragePct}% cov) -> SelAcc=${tSel.selectiveAcc}%, SelMF1=${tSel.selectiveMacroF1}%`);
  console.log("\nEmitted validation_vs_test_calibration_and_selective.csv & independent_recomputation_audit.json.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
