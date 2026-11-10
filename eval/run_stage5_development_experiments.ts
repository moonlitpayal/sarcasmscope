/**
 * SARcasmScope — Stage 5 Development Experiments (Validation Split Only)
 */

import * as fs from "fs";
import * as path from "path";
import { predict as predictLinguistic } from "../src/engine/sarcasmEngine";
import { scoreWordModel, scoreCharModel, predictStatistical } from "../src/engine/statisticalModel";
import {
  predictHybrid,
  DEFAULT_FUSION_WEIGHTS,
  DEFAULT_ABSTAIN_THRESHOLD,
  HybridAblationOptions,
} from "../src/engine/hybridFusion";
import { calculateAllMetrics, DataRecord } from "./run_stage9_master_execution";

export async function runStage5DevelopmentExperiments() {
  const valRecords: DataRecord[] = fs
    .readFileSync(path.resolve("./eval_results/independent/val.jsonl"), "utf8")
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l));
  const valLabels = valRecords.map((r) => r.binary_sarcasm_label);

  const stage5Dir = path.resolve("./eval_results/stage5");
  fs.mkdirSync(stage5Dir, { recursive: true });

  const probsLing = valRecords.map((r) => predictLinguistic(r.text, r.context).p_sarcastic);
  const probsWord = valRecords.map((r) => scoreWordModel(r.text, r.context).prob);
  const probsChar = valRecords.map((r) => scoreCharModel(r.text, r.context).prob);
  const probsStat = valRecords.map((r) => predictStatistical(r.text, r.context).p_statistical);
  const hybRes = valRecords.map((r) => predictHybrid(r.text, r.context));
  const probsHyb = hybRes.map((h) => h.p_fused);

  const mLing = calculateAllMetrics(probsLing, valLabels);
  const mWord = calculateAllMetrics(probsWord, valLabels);
  const mChar = calculateAllMetrics(probsChar, valLabels);
  const mStat = calculateAllMetrics(probsStat, valLabels);
  const mHyb = calculateAllMetrics(probsHyb, valLabels);

  const row = (id: string, rep: string, m: ReturnType<typeof calculateAllMetrics>) =>
    `"${id}","${rep}",${m.accuracy},${m.precision},${m.recall},${m.macroF1},${m.specificity},${m.confidenceECE},${m.classECE},${m.brierScore},${m.nll}`;

  fs.writeFileSync(
    path.join(stage5Dir, "model_comparison.csv"),
    [
      "ModelArchitecture,Representation,Accuracy,Precision,Recall,MacroF1,Specificity,ConfidenceECE,ClassECE,BrierScore,NLL",
      row("Linguistic_Only_v2", "Symbolic Lexicon + Indic Morph + Phonetic", mLing),
      row("WordLex_1_2", "Word + Morph + Phonetic + Ctx TF-IDF LR", mWord),
      row("CharSub_3_5", "Character 3-5 Shingles TF-IDF LR", mChar),
      row("Statistical_OOF_Ensemble", "Logit Mean (WordLex + CharSub)", mStat),
      row("Calibrated_Hybrid_Fusion", "3-Stage OOF MAP Ridge Fusion (T*=0.845)", mHyb),
    ].join("\n") + "\n"
  );

  const abSpecs: { name: string; opts: HybridAblationOptions }[] = [
    { name: "Full Hybrid Model", opts: {} },
    { name: "w/o WordLex Expert", opts: { useWord: false } },
    { name: "w/o CharSub Expert", opts: { useChar: false } },
    { name: "w/o Linguistic Engine Logit", opts: { useLinguistic: false } },
    { name: "w/o Incongruity Index Feature", opts: { useIncongruity: false } },
    { name: "w/o Lexical Coverage Feature", opts: { useCoverage: false } },
    { name: "w/o Conversational Context", opts: { useContext: false } },
    { name: "w/o Temperature Scaling (T=1.0)", opts: { useCalibration: false } },
  ];
  fs.writeFileSync(
    path.join(stage5Dir, "ablation_val_analysis.csv"),
    [
      "AblationConfiguration,Accuracy,Precision,Recall,MacroF1,ConfidenceECE,NLL,DeltaMacroF1",
      ...abSpecs.map((s) => {
        const p = valRecords.map((r) => predictHybrid(r.text, r.context, undefined, DEFAULT_FUSION_WEIGHTS, DEFAULT_ABSTAIN_THRESHOLD, s.opts).p_fused);
        const m = calculateAllMetrics(p, valLabels);
        return `"${s.name}",${m.accuracy},${m.precision},${m.recall},${m.macroF1},${m.confidenceECE},${m.nll},${(m.macroF1 - mHyb.macroF1).toFixed(2)}`;
      }),
    ].join("\n") + "\n"
  );

  const byLang: Record<string, { p: number[]; y: number[] }> = {};
  const byDs: Record<string, { p: number[]; y: number[] }> = {};
  valRecords.forEach((r, i) => {
    (byLang[r.language] ??= { p: [], y: [] }).p.push(probsHyb[i]);
    byLang[r.language].y.push(valLabels[i]);
    (byDs[r.dataset] ??= { p: [], y: [] }).p.push(probsHyb[i]);
    byDs[r.dataset].y.push(valLabels[i]);
  });

  fs.writeFileSync(
    path.join(stage5Dir, "language_error_analysis.csv"),
    [
      "Language,SampleCount,SarcasticCount,LiteralCount,Accuracy,Precision,Recall,MacroF1,Specificity,ConfidenceECE",
      ...Object.entries(byLang).map(([lang, d]) => {
        const m = calculateAllMetrics(d.p, d.y);
        const s = d.y.reduce((a, b) => a + b, 0);
        return `"${lang}",${d.y.length},${s},${d.y.length - s},${m.accuracy},${m.precision},${m.recall},${m.macroF1},${m.specificity},${m.confidenceECE}`;
      }),
    ].join("\n") + "\n"
  );

  fs.writeFileSync(
    path.join(stage5Dir, "domain_error_analysis.csv"),
    [
      "Dataset,SampleCount,Accuracy,Precision,Recall,MacroF1,Specificity",
      ...Object.entries(byDs).map(([ds, d]) => {
        const m = calculateAllMetrics(d.p, d.y);
        return `"${ds}",${d.y.length},${m.accuracy},${m.precision},${m.recall},${m.macroF1},${m.specificity}`;
      }),
    ].join("\n") + "\n"
  );

  fs.writeFileSync(
    path.join(stage5Dir, "val_predictions.csv"),
    [
      "ID,Dataset,Language,GroundTruth,ProbHybrid,PredHybrid,Confidence,Abstain,AbstainReason",
      ...valRecords.map((r, i) => {
        const h = hybRes[i];
        return `"${r.id}","${r.dataset}","${r.language}",${r.binary_sarcasm_label},${h.p_fused},${h.p_fused >= 0.5 ? 1 : 0},${h.confidence},${h.abstain ? 1 : 0},"${h.abstain_reason || "None"}"`;
      }),
    ].join("\n") + "\n"
  );

  console.log(`Stage 5 validation CSVs saved cleanly to ${stage5Dir}`);
}

const isMain =
  typeof process !== "undefined" &&
  process.argv[1] &&
  (process.argv[1].endsWith("run_stage5_development_experiments.ts") ||
    process.argv[1].endsWith("run_stage5_development_experiments.js"));

if (isMain) {
  runStage5DevelopmentExperiments().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
