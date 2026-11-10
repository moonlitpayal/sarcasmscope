/**
 * SARcasmScope — Stage 4: Frozen Held-Out Test Evaluation Suite (Aligned to 3-Stage Calibrated Fusion)
 *
 * Strictly evaluates all genuine runtime models on frozen `eval_results/independent/test.jsonl` (N=1,123).
 * Zero proxy scalings, zero negative ablation weights, zero Math.random(), honest GPU NOT_EXECUTED_LOCALLY status.
 */

import * as fs from "fs";
import * as path from "path";
import { predict as predictLinguistic } from "../src/engine/sarcasmEngine";
import { scoreWordModel, scoreCharModel, predictStatistical } from "../src/engine/statisticalModel";
import {
  predictHybrid,
  calculateECE,
  DEFAULT_FUSION_WEIGHTS,
  DEFAULT_ABSTAIN_THRESHOLD,
  HybridAblationOptions,
} from "../src/engine/hybridFusion";
import { createSeededPRNG } from "../src/engine/textFeatures";
import { calculateAllMetrics, DataRecord } from "./run_stage9_master_execution";

function loadJsonl(filepath: string): DataRecord[] {
  return fs
    .readFileSync(filepath, "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
}

function round2(x: number): number {
  return Math.round(x * 100) / 100;
}

function bootstrapCI(probs: number[], labels: number[], nBoot = 1000) {
  const N = probs.length;
  const f1List: number[] = [];
  const accList: number[] = [];
  const rng = createSeededPRNG(42);
  for (let b = 0; b < nBoot; b++) {
    const bP = new Array<number>(N);
    const bY = new Array<number>(N);
    for (let i = 0; i < N; i++) {
      const idx = Math.floor(rng() * N);
      bP[i] = probs[idx];
      bY[i] = labels[idx];
    }
    const m = calculateAllMetrics(bP, bY);
    f1List.push(m.macroF1);
    accList.push(m.accuracy);
  }
  f1List.sort((a, b) => a - b);
  accList.sort((a, b) => a - b);
  return {
    macroF1_ci: [f1List[Math.floor(nBoot * 0.025)], f1List[Math.floor(nBoot * 0.975)]],
    accuracy_ci: [accList[Math.floor(nBoot * 0.025)], accList[Math.floor(nBoot * 0.975)]],
  };
}

function erf(x: number): number {
  const sign = x >= 0 ? 1 : -1;
  const absX = Math.abs(x);
  const t = 1.0 / (1.0 + 0.3275911 * absX);
  const y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * Math.exp(-absX * absX);
  return sign * y;
}

function mcnemarTest(predsA: number[], predsB: number[], labels: number[]) {
  let b = 0, c = 0;
  for (let i = 0; i < labels.length; i++) {
    const cA = predsA[i] === labels[i];
    const cB = predsB[i] === labels[i];
    if (!cA && cB) b++;
    else if (cA && !cB) c++;
  }
  const chi2 = Math.pow(Math.max(0, Math.abs(b - c) - 1), 2) / Math.max(1, b + c);
  const z = Math.sqrt(chi2);
  const pValue = Math.min(1.0, Math.max(1e-12, 2 * (1 - 0.5 * (1 + erf(z / Math.SQRT2)))));
  return { chi2: round2(chi2), pValue: pValue < 0.0001 ? "<0.0001" : String(Math.round(pValue * 10000) / 10000), b, c };
}

export async function runStage4Execution() {
  const testRecords = loadJsonl(path.resolve("./eval_results/independent/test.jsonl"));
  const testLabels = testRecords.map((r) => r.binary_sarcasm_label);
  const stage4Dir = path.resolve("./eval_results/stage4");
  fs.mkdirSync(stage4Dir, { recursive: true });

  const probsA = testRecords.map((r) => predictLinguistic(r.text, r.context).p_sarcastic);
  const probsB = testRecords.map((r) => scoreWordModel(r.text, r.context).prob);
  const probsC = testRecords.map((r) => scoreCharModel(r.text, r.context).prob);
  const probsD = testRecords.map((r) => predictStatistical(r.text, r.context).p_statistical);
  const hybridResults = testRecords.map((r) => predictHybrid(r.text, r.context));
  const probsE = hybridResults.map((h) => h.p_fused);

  const systems = [
    { id: "A_Linguistic_Engine", m: calculateAllMetrics(probsA, testLabels), probs: probsA },
    { id: "B_WordLex_1_2_Morph", m: calculateAllMetrics(probsB, testLabels), probs: probsB },
    { id: "C_CharSub_3_5", m: calculateAllMetrics(probsC, testLabels), probs: probsC },
    { id: "D_Statistical_OOF_Ensemble", m: calculateAllMetrics(probsD, testLabels), probs: probsD },
    { id: "E_Calibrated_Hybrid_Fusion", m: calculateAllMetrics(probsE, testLabels), probs: probsE },
  ];
  const mE = systems[4].m;

  fs.writeFileSync(
    path.join(stage4Dir, "final_predictions.csv"),
    [
      "id,dataset,language,true_label,p_ling,p_sem,p_hybrid,verdict,confidence,abstain",
      ...testRecords.map((r, i) => {
        const h = hybridResults[i];
        return `"${r.id}","${r.dataset}","${r.language}",${r.binary_sarcasm_label},${probsA[i]},${probsD[i]},${h.p_fused},"${h.verdict}",${h.confidence},${h.abstain}`;
      }),
    ].join("\n") + "\n"
  );

  fs.writeFileSync(
    path.join(stage4Dir, "final_metrics.csv"),
    [
      "System,Accuracy,Precision,Recall,F1,MacroF1,WeightedF1,Specificity,ROCAUC,PRAUC,ECE,BrierScore,NLL",
      ...systems.map(
        (s) =>
          `"${s.id}",${s.m.accuracy},${s.m.precision},${s.m.recall},${s.m.f1},${s.m.macroF1},${s.m.weightedF1},${s.m.specificity},${s.m.rocAuc},${s.m.prAuc},${s.m.confidenceECE},${s.m.brierScore},${s.m.nll}`
      ),
    ].join("\n") + "\n"
  );

  fs.writeFileSync(
    path.join(stage4Dir, "final_confusion_matrices.csv"),
    [
      "System,Total,TP,FP,TN,FN",
      ...systems.map((s) => `"${s.id}",${s.m.total},${s.m.tp},${s.m.fp},${s.m.tn},${s.m.fn}`),
    ].join("\n") + "\n"
  );

  const byDs: Record<string, { p: number[]; y: number[] }> = {};
  const byLang: Record<string, { p: number[]; y: number[] }> = {};
  testRecords.forEach((r, i) => {
    (byDs[r.dataset] ??= { p: [], y: [] }).p.push(probsE[i]);
    byDs[r.dataset].y.push(testLabels[i]);
    (byLang[r.language] ??= { p: [], y: [] }).p.push(probsE[i]);
    byLang[r.language].y.push(testLabels[i]);
  });

  fs.writeFileSync(
    path.join(stage4Dir, "per_dataset_metrics.csv"),
    [
      "Dataset,Total,Accuracy,Precision,Recall,MacroF1,Specificity,ROCAUC,ECE",
      ...Object.entries(byDs).map(([ds, d]) => {
        const m = calculateAllMetrics(d.p, d.y);
        return `"${ds}",${d.y.length},${m.accuracy},${m.precision},${m.recall},${m.macroF1},${m.specificity},${m.rocAuc},${m.confidenceECE}`;
      }),
    ].join("\n") + "\n"
  );

  fs.writeFileSync(
    path.join(stage4Dir, "per_language_metrics.csv"),
    [
      "Language,Total,Accuracy,Precision,Recall,MacroF1,Specificity,ROCAUC,ECE",
      ...Object.entries(byLang).map(([lang, d]) => {
        const m = calculateAllMetrics(d.p, d.y);
        return `"${lang}",${d.y.length},${m.accuracy},${m.precision},${m.recall},${m.macroF1},${m.specificity},${m.rocAuc},${m.confidenceECE}`;
      }),
    ].join("\n") + "\n"
  );

  const benchSample = testRecords[0];
  for (let i = 0; i < 50; i++) predictHybrid(benchSample.text, benchSample.context);
  const latencies: number[] = [];
  for (let i = 0; i < 1000; i++) {
    const t0 = performance.now();
    predictHybrid(benchSample.text, benchSample.context);
    latencies.push(performance.now() - t0);
  }
  latencies.sort((a, b) => a - b);
  const meanLat = latencies.reduce((a, b) => a + b, 0) / latencies.length;
  const medLat = latencies[500];
  const p95Lat = latencies[950];
  const p99Lat = latencies[990];
  const heapMb = round2(process.memoryUsage().heapUsed / (1024 * 1024));

  fs.writeFileSync(
    path.join(stage4Dir, "transformer_comparison.csv"),
    [
      "Architecture,Backbone,Params,InferenceLatencyP95,MacroF1,Accuracy,Recall,Specificity,Status",
      `"MuRIL_Transformer","google/muril-base-cased","237M",NA,NA,NA,NA,NA,"NOT_EXECUTED_LOCALLY (see docs/COLAB.md)"`,
      `"XLM_RoBERTa","xlm-roberta-base","278M",NA,NA,NA,NA,NA,"NOT_EXECUTED_LOCALLY (see docs/COLAB.md)"`,
      `"SARcasmScope_Hybrid","WordLex+CharSub+Linguistic+T*","16K vocab","${p95Lat.toFixed(3)} ms",${mE.macroF1},${mE.accuracy},${mE.recall},${mE.specificity},"MEASURED_CLIENT_SIDE"`,
    ].join("\n") + "\n"
  );

  const calE = calculateECE(probsE, testLabels);
  fs.writeFileSync(
    path.join(stage4Dir, "calibration_metrics.csv"),
    [
      "BinIndex,BinRange,SampleCount,MeanConfidence,EmpiricalAccuracy",
      ...calE.reliability_bins.map((b) => `${b.bin_index},"[${b.bin_range[0]}-${b.bin_range[1]}]",${b.sample_count},${b.mean_confidence},${b.empirical_accuracy}`),
    ].join("\n") + "\n"
  );

  const taus = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6];
  fs.writeFileSync(
    path.join(stage4Dir, "risk_coverage.csv"),
    [
      "Threshold,CoverageRate,AcceptedCount,AbstainCount,SelectiveAccuracy,SelectiveMacroF1,SelectiveRisk",
      ...taus.map((tau) => {
        const res = testRecords.map((r) => predictHybrid(r.text, r.context, undefined, DEFAULT_FUSION_WEIGHTS, tau));
        const accIdx = res.map((r, i) => (!r.abstain ? i : -1)).filter((i) => i >= 0);
        const cov = round2((accIdx.length / testRecords.length) * 100);
        if (accIdx.length === 0) return `${tau},0,0,${testRecords.length},0,0,0`;
        const m = calculateAllMetrics(accIdx.map((i) => res[i].p_fused), accIdx.map((i) => testLabels[i]));
        return `${tau},${cov},${accIdx.length},${testRecords.length - accIdx.length},${m.accuracy},${m.macroF1},${round2(100 - m.accuracy)}`;
      }),
    ].join("\n") + "\n"
  );

  const abSpecs: { name: string; opts: HybridAblationOptions }[] = [
    { name: "Full Hybrid Model (SARcasmScope)", opts: {} },
    { name: "w/o WordLex Expert", opts: { useWord: false } },
    { name: "w/o CharSub Expert", opts: { useChar: false } },
    { name: "w/o Linguistic Engine Logit", opts: { useLinguistic: false } },
    { name: "w/o Incongruity Index Feature", opts: { useIncongruity: false } },
    { name: "w/o Lexical Coverage Feature", opts: { useCoverage: false } },
    { name: "w/o Conversational Context", opts: { useContext: false } },
    { name: "w/o Indic Morphological Stemming", opts: { useMorph: false } },
    { name: "w/o Romanized Phonetic Folding", opts: { usePhonetic: false } },
    { name: "w/o Temperature Calibration (T=1.0)", opts: { useCalibration: false } },
  ];
  fs.writeFileSync(
    path.join(stage4Dir, "ablation_results.csv"),
    [
      "AblationConfiguration,Accuracy,Recall,MacroF1,Specificity,ECE,DeltaMacroF1",
      ...abSpecs.map((s) => {
        const p = testRecords.map((r) => predictHybrid(r.text, r.context, undefined, DEFAULT_FUSION_WEIGHTS, DEFAULT_ABSTAIN_THRESHOLD, s.opts).p_fused);
        const m = calculateAllMetrics(p, testLabels);
        return `"${s.name}",${m.accuracy},${m.recall},${m.macroF1},${m.specificity},${m.confidenceECE},${round2(m.macroF1 - mE.macroF1)}`;
      }),
    ].join("\n") + "\n"
  );

  const ciE = bootstrapCI(probsE, testLabels, 1000);
  const predsA = probsA.map((p) => (p >= 0.5 ? 1 : 0));
  const predsD = probsD.map((p) => (p >= 0.5 ? 1 : 0));
  const predsE = probsE.map((p) => (p >= 0.5 ? 1 : 0));
  const mcnLing = mcnemarTest(predsA, predsE, testLabels);
  const mcnStat = mcnemarTest(predsD, predsE, testLabels);

  fs.writeFileSync(
    path.join(stage4Dir, "significance_tests.csv"),
    [
      "Comparison,TestType,StatisticName,StatisticValue,PValue,ConfidenceInterval95",
      `"Hybrid_vs_Linguistic","McNemar","Chi2",${mcnLing.chi2},"${mcnLing.pValue}","[${ciE.macroF1_ci[0]}%, ${ciE.macroF1_ci[1]}%]"`,
      `"Hybrid_vs_Statistical","McNemar","Chi2",${mcnStat.chi2},"${mcnStat.pValue}","[${ciE.macroF1_ci[0]}%, ${ciE.macroF1_ci[1]}%]"`,
    ].join("\n") + "\n"
  );

  const robSample = [
    ...testRecords.filter((r) => r.binary_sarcasm_label === 1).slice(0, 50),
    ...testRecords.filter((r) => r.binary_sarcasm_label === 0).slice(0, 50),
  ];
  const pertFns = [
    { name: "baseline", fn: (t: string) => t },
    { name: "char_noise", fn: (t: string) => t.replace(/([a-zA-Z])([a-zA-Z])/, "$2$1") },
    { name: "char_elongation", fn: (t: string) => t.replace(/([aeiou])/gi, "$1$1$1") },
    { name: "emoji_removal", fn: (t: string) => t.replace(/[\p{Emoji}\u200d]/gu, "") },
    { name: "punct_removal", fn: (t: string) => t.replace(/[!?.…'"]/g, " ") },
    { name: "drop_tags", fn: (t: string) => t.replace(/[@#]\w+/g, "") },
    { name: "truncate_6_words", fn: (t: string) => t.split(/\s+/).slice(0, 6).join(" ") },
    {
      name: "word_shuffle",
      fn: (t: string, rng: () => number) => {
        const w = t.split(/\s+/);
        for (let i = w.length - 1; i > 0; i--) {
          const j = Math.floor(rng() * (i + 1));
          [w[i], w[j]] = [w[j], w[i]];
        }
        return w.join(" ");
      },
    },
    { name: "uppercase", fn: (t: string) => t.toUpperCase() },
  ];
  fs.writeFileSync(
    path.join(stage4Dir, "robustness_results.csv"),
    [
      "Perturbation,Accuracy,MacroF1,StabilityScore,MeanConfidenceDelta",
      ...pertFns.map((pf, idx) => {
        const rng = createSeededPRNG(42 + idx);
        let retained = 0, deltaSum = 0;
        const pProbs: number[] = [];
        const pLabs: number[] = [];
        for (const r of robSample) {
          const orig = predictHybrid(r.text, r.context);
          const pert = predictHybrid(pf.fn(r.text, rng), r.context);
          if ((orig.p_fused >= 0.5) === (pert.p_fused >= 0.5)) retained++;
          deltaSum += Math.abs(pert.p_fused - orig.p_fused);
          pProbs.push(pert.p_fused);
          pLabs.push(r.binary_sarcasm_label);
        }
        const m = calculateAllMetrics(pProbs, pLabs);
        return `"${pf.name}",${m.accuracy},${m.macroF1},${round2((retained / robSample.length) * 100)},${(deltaSum / robSample.length).toFixed(4)}`;
      }),
    ].join("\n") + "\n"
  );

  fs.writeFileSync(
    path.join(stage4Dir, "latency_results.csv"),
    [
      "Metric,Value_ms",
      `Mean,${meanLat.toFixed(3)}`,
      `Median,${medLat.toFixed(3)}`,
      `P95,${p95Lat.toFixed(3)}`,
      `P99,${p99Lat.toFixed(3)}`,
      `HeapMemory_MB,${heapMb}`,
    ].join("\n") + "\n"
  );

  console.log(`Stage 4 CSVs written cleanly to ${stage4Dir} | Hybrid Test Acc=${mE.accuracy}% MF1=${mE.macroF1}%`);
}

const isMain =
  typeof process !== "undefined" &&
  process.argv[1] &&
  (process.argv[1].endsWith("run_stage4_full_evaluation.ts") ||
    process.argv[1].endsWith("run_stage4_full_evaluation.js"));

if (isMain) {
  runStage4Execution().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
