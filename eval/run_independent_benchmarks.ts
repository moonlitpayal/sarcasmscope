/**
 * SARcasmScope Reconciled Independent Held-Out Benchmark Runner
 *
 * Evaluates frozen `eval_results/independent/test.jsonl` (N = 1,123 across 5 corpora:
 * 344 ACL 2020 Twitter + 401 Misra Headlines + 194 IIITD MaSaC + 75 L3Cube MahaEmotions + 109 IIIT-H Telugu)
 * using the canonical 3-Stage Calibrated Feature-Fusion Engine (`predictHybrid`) alongside the
 * Stage 2 Symbolic Linguistic Engine (`predict`).
 *
 * Strictly zero parameter fitting and zero split modification occurs here.
 * Emits synchronized CSVs inside `eval_results/independent/` matching `eval_results/stage9/master_summary.json`.
 */

import * as fs from "fs";
import * as path from "path";
import { predict as predictLinguistic } from "../src/engine/sarcasmEngine";
import { predictHybrid, calculateECE } from "../src/engine/hybridFusion";

interface FrozenTestRecord {
  id: string;
  dataset: string;
  language: string;
  domain: string;
  text: string;
  context: string;
  conversation_id?: string;
  binary_sarcasm_label: 0 | 1;
}

interface DatasetRow {
  datasetId: string;
  datasetName: string;
  domain: string;
  language: string;
  hasContext: boolean;
  n: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  macroF1: number;
  weightedF1: number;
  confEce: number;
  brier: number;
  nll: number;
  ciLower: number;
  ciUpper: number;
  avgLatencyMs: number;
  lingAccuracy: number;
  lingMacroF1: number;
  lingRecall: number;
}

function loadFrozenTest(): FrozenTestRecord[] {
  const p = path.resolve("./eval_results/independent/test.jsonl");
  return fs
    .readFileSync(p, "utf8")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function computeMetrics(
  yTrue: number[],
  probs: number[],
  lingProbs: number[],
  latencies: number[],
  meta: { datasetId: string; datasetName: string; domain: string; language: string; hasContext: boolean }
): DatasetRow {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  let ltp = 0, lfp = 0, ltn = 0, lfn = 0;
  const n = yTrue.length;

  for (let i = 0; i < n; i++) {
    const p = probs[i] >= 0.5 ? 1 : 0;
    const lp = lingProbs[i] >= 0.5 ? 1 : 0;
    const y = yTrue[i];
    if (p === 1 && y === 1) tp++;
    else if (p === 1 && y === 0) fp++;
    else if (p === 0 && y === 0) tn++;
    else fn++;

    if (lp === 1 && y === 1) ltp++;
    else if (lp === 1 && y === 0) lfp++;
    else if (lp === 0 && y === 0) ltn++;
    else lfn++;
  }

  const n1 = yTrue.filter((y) => y === 1).length;
  const n0 = n - n1;

  const accuracy = n > 0 ? (tp + tn) / n : 0;
  const prec1 = tp + fp > 0 ? tp / (tp + fp) : 0;
  const rec1 = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1_1 = prec1 + rec1 > 0 ? (2 * prec1 * rec1) / (prec1 + rec1) : 0;
  const prec0 = tn + fn > 0 ? tn / (tn + fn) : 0;
  const rec0 = tn + fp > 0 ? tn / (tn + fp) : 0;
  const f1_0 = prec0 + rec0 > 0 ? (2 * prec0 * rec0) / (prec0 + rec0) : 0;
  const macroF1 = (f1_1 + f1_0) / 2;
  const weightedF1 = n > 0 ? (f1_1 * n1 + f1_0 * n0) / n : 0;

  const lAcc = n > 0 ? ((ltp + ltn) / n) * 100 : 0;
  const lp1 = ltp + lfp > 0 ? ltp / (ltp + lfp) : 0;
  const lr1 = ltp + lfn > 0 ? ltp / (ltp + lfn) : 0;
  const lf1 = lp1 + lr1 > 0 ? (2 * lp1 * lr1) / (lp1 + lr1) : 0;
  const lp0 = ltn + lfn > 0 ? ltn / (ltn + lfn) : 0;
  const lr0 = ltn + lfp > 0 ? ltn / (ltn + lfp) : 0;
  const lf0 = lp0 + lr0 > 0 ? (2 * lp0 * lr0) / (lp0 + lr0) : 0;
  const lMacroF1 = ((lf1 + lf0) / 2) * 100;

  // Wilson 95% CI
  const z = 1.96;
  const denom = 1 + (z * z) / n;
  const center = accuracy + (z * z) / (2 * n);
  const std = Math.sqrt((accuracy * (1 - accuracy) + (z * z) / (4 * n)) / n);
  const ciLower = Math.max(0, (center - z * std) / denom) * 100;
  const ciUpper = Math.min(1, (center + z * std) / denom) * 100;

  const cal = calculateECE(probs, yTrue);
  const avgLat = latencies.reduce((a, b) => a + b, 0) / Math.max(1, latencies.length);

  return {
    ...meta,
    n,
    tp,
    fp,
    tn,
    fn,
    accuracy: accuracy * 100,
    precision: prec1 * 100,
    recall: rec1 * 100,
    f1: f1_1 * 100,
    macroF1: macroF1 * 100,
    weightedF1: weightedF1 * 100,
    confEce: cal.ece,
    brier: cal.brier_score,
    nll: cal.nll,
    ciLower,
    ciUpper,
    avgLatencyMs: avgLat,
    lingAccuracy: lAcc,
    lingMacroF1: lMacroF1,
    lingRecall: lr1 * 100,
  };
}

async function run() {
  const indepDir = path.resolve("./eval_results/independent");
  const testRecords = loadFrozenTest();

  const DATASET_META: Record<
    string,
    { name: string; domain: string; hasContext: boolean; failureNote: string }
  > = {
    ACL_2020_Twitter_Sarcasm: {
      name: "ACL 2020 Twitter Sarcasm",
      domain: "Conversational Social Media Dialogue (Twitter)",
      hasContext: true,
      failureNote: "Implicit world knowledge, hashtag irony, pragmatic ellipsis",
    },
    Misra_Arora_News_Headlines: {
      name: "Misra-Arora News Headlines",
      domain: "Editorial News Satire vs Journalism (No Dialogue Context)",
      hasContext: false,
      failureNote: "Genre difference: editorial satire without multi-turn dialogue",
    },
    IIITD_MaSaC_CodeMixed: {
      name: "IIITD MaSaC Code-Mixed",
      domain: "Isolated Sitcom Utterances (No Dialogue Text / No A/V Prosody)",
      hasContext: false,
      failureNote: "Isolated text turns (context='', conv_id split-grouped only); missing audio/facial prosody; 5 train-test 3-shingle near-dups",
    },
    L3Cube_MahaEmotions: {
      name: "L3Cube-MahaEmotions Marathi",
      domain: "Restricted Emotion-Proxy Twitter Subset (Sarcasm vs Neutral Only)",
      hasContext: false,
      failureNote: "Restricted label proxy (Sarcasm vs Neutral excludes non-neutral emotional literal tweets)",
    },
    IIIT_Hyderabad_Telugu_Sarcasm: {
      name: "IIIT-H Telugu Sarcasm",
      domain: "Native Telugu Social Media (15.60% Natural Sarcasm Skew)",
      hasContext: false,
      failureNote: "0% Hybrid recall at global 0.50 threshold under minority prior despite 0.7455 ROC-AUC",
    },
  };

  // Warmup
  for (let i = 0; i < Math.min(20, testRecords.length); i++) {
    predictHybrid(testRecords[i].text, testRecords[i].context);
  }

  const byDataset = new Map<
    string,
    { yTrue: number[]; probs: number[]; lingProbs: number[]; latencies: number[]; language: string }
  >();
  const sampleRows: string[] = [
    "ID,Dataset,Language,GroundTruth,Hybrid_P_Fused,Hybrid_Pred,Linguistic_P,Confidence_Margin,Abstain,Correct,Text",
  ];

  const allY: number[] = [];
  const allP: number[] = [];
  const allLP: number[] = [];
  const allLat: number[] = [];

  for (const r of testRecords) {
    const t0 = performance.now();
    const hyb = predictHybrid(r.text, r.context);
    const dt = performance.now() - t0;
    const ling = predictLinguistic(r.text, r.context);

    const predLabel = hyb.p_fused >= 0.5 ? 1 : 0;
    allY.push(r.binary_sarcasm_label);
    allP.push(hyb.p_fused);
    allLP.push(ling.p_sarcastic);
    allLat.push(dt);

    if (!byDataset.has(r.dataset)) {
      byDataset.set(r.dataset, {
        yTrue: [],
        probs: [],
        lingProbs: [],
        latencies: [],
        language: r.language,
      });
    }
    const bucket = byDataset.get(r.dataset)!;
    bucket.yTrue.push(r.binary_sarcasm_label);
    bucket.probs.push(hyb.p_fused);
    bucket.lingProbs.push(ling.p_sarcastic);
    bucket.latencies.push(dt);

    const safeText = r.text.replace(/[\r\n\t"]/g, " ").slice(0, 120);
    sampleRows.push(
      `${r.id},"${r.dataset}","${r.language}",${r.binary_sarcasm_label},${hyb.p_fused.toFixed(4)},${predLabel},${ling.p_sarcastic.toFixed(4)},${hyb.confidence.toFixed(4)},${hyb.abstain},"${predLabel === r.binary_sarcasm_label ? 1 : 0}","${safeText}"`
    );
  }

  const orderedIds = [
    "ACL_2020_Twitter_Sarcasm",
    "Misra_Arora_News_Headlines",
    "IIITD_MaSaC_CodeMixed",
    "L3Cube_MahaEmotions",
    "IIIT_Hyderabad_Telugu_Sarcasm",
  ];

  const rows: DatasetRow[] = orderedIds.map((id) => {
    const b = byDataset.get(id)!;
    const m = DATASET_META[id];
    return computeMetrics(b.yTrue, b.probs, b.lingProbs, b.latencies, {
      datasetId: id,
      datasetName: m.name,
      domain: m.domain,
      language: b.language,
      hasContext: m.hasContext,
    });
  });

  const overall = computeMetrics(allY, allP, allLP, allLat, {
    datasetId: "ALL_TEST_1123",
    datasetName: "ALL HELD-OUT TEST (5 Corpora Aggregate)",
    domain: "Multilingual & Multi-Domain Held-Out Split",
    language: "Multilingual",
    hasContext: true,
  });

  const metricHeader =
    "Dataset,Domain,Language,HasContext,N,TP,FP,TN,FN,Accuracy,Precision,Recall,F1_Sarcasm,Macro_F1,Weighted_F1,Conf_ECE,BrierScore,NLL,95CI_Lower,95CI_Upper,AvgLatency_ms,Linguistic_v2_Acc,Linguistic_v2_MacroF1,Linguistic_v2_Recall\n";
  const formatRow = (r: DatasetRow) =>
    [
      `"${r.datasetName}"`,
      `"${r.domain}"`,
      `"${r.language}"`,
      r.hasContext,
      r.n,
      r.tp,
      r.fp,
      r.tn,
      r.fn,
      r.accuracy.toFixed(2),
      r.precision.toFixed(2),
      r.recall.toFixed(2),
      r.f1.toFixed(2),
      r.macroF1.toFixed(2),
      r.weightedF1.toFixed(2),
      r.confEce.toFixed(2),
      r.brier.toFixed(4),
      r.nll.toFixed(4),
      r.ciLower.toFixed(2),
      r.ciUpper.toFixed(2),
      r.avgLatencyMs.toFixed(4),
      r.lingAccuracy.toFixed(2),
      r.lingMacroF1.toFixed(2),
      r.lingRecall.toFixed(2),
    ].join(",");

  fs.writeFileSync(
    path.join(indepDir, "independent_evaluation_metrics.csv"),
    metricHeader + [...rows, overall].map(formatRow).join("\n") + "\n"
  );

  const langHeader =
    "Language,EvaluatedDatasets,TotalTestSamples,TruePositives,FalsePositives,TrueNegatives,FalseNegatives,Precision,Recall,F1_Sarcasm,Macro_F1,Conf_ECE\n";
  const langRows = [
    ...rows.map((r) =>
      [
        `"${r.language === "English" ? (r.datasetId.includes("Twitter") ? "English (Conversational)" : "English (Headlines)") : r.language}"`,
        `"${r.datasetName}"`,
        r.n,
        r.tp,
        r.fp,
        r.tn,
        r.fn,
        r.precision.toFixed(2),
        r.recall.toFixed(2),
        r.f1.toFixed(2),
        r.macroF1.toFixed(2),
        r.confEce.toFixed(2),
      ].join(",")
    ),
    [
      `"ALL TEST (Aggregate)"`,
      `"5 Frozen Test Partitions"`,
      overall.n,
      overall.tp,
      overall.fp,
      overall.tn,
      overall.fn,
      overall.precision.toFixed(2),
      overall.recall.toFixed(2),
      overall.f1.toFixed(2),
      overall.macroF1.toFixed(2),
      overall.confEce.toFixed(2),
    ].join(","),
  ];
  fs.writeFileSync(
    path.join(indepDir, "language_wise_independent_performance.csv"),
    langHeader + langRows.join("\n") + "\n"
  );

  const crossHeader =
    "SourceTrainingSplit,TargetTestDataset,HeldOutTestSamples,MacroF1,Accuracy,Recall,Precision,DomainValidityQualification\n";
  const crossRows = rows.map((r) =>
    [
      `"train.jsonl (N=3,373, Val T*=0.845)"`,
      `"${r.domain}"`,
      r.n,
      r.macroF1.toFixed(2),
      r.accuracy.toFixed(2),
      r.recall.toFixed(2),
      r.precision.toFixed(2),
      `"${DATASET_META[r.datasetId].failureNote}"`,
    ].join(",")
  );
  fs.writeFileSync(
    path.join(indepDir, "cross_dataset_domain_generalization.csv"),
    crossHeader + crossRows.join("\n") + "\n"
  );

  fs.writeFileSync(
    path.join(indepDir, "independent_sample_predictions.csv"),
    sampleRows.join("\n") + "\n"
  );

  console.log(
    `Synchronized eval_results/independent/*.csv to frozen test.jsonl (N=${overall.n}: Acc=${overall.accuracy.toFixed(2)}%, MF1=${overall.macroF1.toFixed(2)}%, Rec=${overall.recall.toFixed(2)}%, ECE=${overall.confEce.toFixed(2)}%).`
  );
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
