/**
 * SARcasmScope — Stage 8 Master Validation Synchronizer
 * Synchronizes eval_results/stage8/ CSVs strictly from live Stage 9 measurements (no hardcoded numbers or Math.random()).
 */

import * as fs from "fs";
import * as path from "path";

export async function runStage8MasterValidation() {
  const stage8Dir = path.resolve("./eval_results/stage8");
  const stage9Dir = path.resolve("./eval_results/stage9");
  fs.mkdirSync(stage8Dir, { recursive: true });

  const master = JSON.parse(fs.readFileSync(path.join(stage9Dir, "master_summary.json"), "utf8"));
  const b30 = master.diagnosticSuites.benchmark30;
  const f = master.frozenTestModels;

  const row = (name: string, m: any) =>
    `"${name}",${m.total},${m.accuracy},${m.precision},${m.recall},${m.macroF1},${m.weightedF1},${m.specificity},${m.rocAuc},${m.prAuc},${m.confidenceECE},${m.classECE},${m.brierScore},${m.nll},${m.tp},${m.fp},${m.tn},${m.fn}`;

  fs.writeFileSync(
    path.join(stage8Dir, "recomputed_metrics.csv"),
    [
      "EvaluationSplit,SampleCount,Accuracy,Precision,Recall,MacroF1,WeightedF1,Specificity,ROCAUC,PRAUC,ConfECE,ClassECE,BrierScore,NLL,TP,FP,TN,FN",
      row("Diagnostic_30_v1_Prev", b30.v1),
      row("Diagnostic_30_v2_Linguistic", b30.v2),
      row("Diagnostic_30_Calibrated_Hybrid", b30.hybrid),
      row("HeldOut_1123_v1_Prev", f.v1PrevSymbolic),
      row("HeldOut_1123_v2_Linguistic", f.v2SymbolicLinguistic),
      row("HeldOut_1123_Statistical_OOF", f.statisticalOOF),
      row("HeldOut_1123_Calibrated_Hybrid", f.calibratedHybrid),
    ].join("\n") + "\n"
  );

  fs.copyFileSync(path.join(stage9Dir, "initial_audit.csv"), path.join(stage8Dir, "reconciliation_report.csv"));
  fs.copyFileSync(path.join(stage9Dir, "dataset_integrity.csv"), path.join(stage8Dir, "dataset_integrity.csv"));
  fs.copyFileSync(path.join(stage9Dir, "model_comparison.csv"), path.join(stage8Dir, "transformer_comparison.csv"));
  fs.copyFileSync(path.join(stage9Dir, "statistical_analysis.csv"), path.join(stage8Dir, "statistical_validation.csv"));
  fs.copyFileSync(path.join(stage9Dir, "calibration_results.csv"), path.join(stage8Dir, "calibration_analysis.csv"));
  fs.copyFileSync(path.join(stage9Dir, "risk_coverage.csv"), path.join(stage8Dir, "risk_coverage.csv"));
  fs.copyFileSync(path.join(stage9Dir, "robustness_results.csv"), path.join(stage8Dir, "robustness_results.csv"));

  fs.copyFileSync(
    path.resolve("./eval_results/stage6/error_analysis.csv"),
    path.join(stage8Dir, "error_analysis.csv")
  );

  console.log(`Stage 8 CSVs synchronized cleanly with Stage 9 measurements in ${stage8Dir}`);
}

const isMain =
  typeof process !== "undefined" &&
  process.argv[1] &&
  (process.argv[1].endsWith("run_stage8_master_validation.ts") ||
    process.argv[1].endsWith("run_stage8_master_validation.js"));

if (isMain) {
  runStage8MasterValidation().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
