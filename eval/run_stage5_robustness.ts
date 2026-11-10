/**
 * SARcasmScope — Stage 5E: Deterministic Validation Robustness Evaluation
 */

import * as fs from "fs";
import * as path from "path";
import { predictHybrid } from "../src/engine/hybridFusion";
import { createSeededPRNG } from "../src/engine/textFeatures";

interface DataRecord {
  id: string;
  dataset: string;
  language: string;
  text: string;
  context: string;
  binary_sarcasm_label: 0 | 1;
}

export async function runRobustnessEvaluation() {
  const valRecords: DataRecord[] = fs
    .readFileSync(path.resolve("./eval_results/independent/val.jsonl"), "utf8")
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l));

  const sample = [
    ...valRecords.filter((r) => r.binary_sarcasm_label === 1).slice(0, 50),
    ...valRecords.filter((r) => r.binary_sarcasm_label === 0).slice(0, 50),
  ];

  const perturbations = [
    {
      name: "Word Shuffling (Permutation)",
      repType: "Order-Invariant Bag-of-Tokens",
      desc: "Deterministically permute word order within target utterance (seed=42)",
      fn: (r: DataRecord, rng: () => number) => {
        const w = r.text.split(/\s+/);
        for (let i = w.length - 1; i > 0; i--) {
          const j = Math.floor(rng() * (i + 1));
          [w[i], w[j]] = [w[j], w[i]];
        }
        return { text: w.join(" "), context: r.context };
      },
    },
    {
      name: "Punctuation Stripping",
      repType: "Syntax / Sequential",
      desc: "Remove exclamation marks, question marks, ellipses, and quotes",
      fn: (r: DataRecord) => ({
        text: r.text.replace(/[!?.…'"]/g, " "),
        context: r.context.replace(/[!?.…'"]/g, " "),
      }),
    },
    {
      name: "Dialogue Context Removal",
      repType: "Syntax / Sequential",
      desc: "Detach preceding conversational context turn",
      fn: (r: DataRecord) => ({ text: r.text, context: "" }),
    },
    {
      name: "Typographical Noise (Character Swap)",
      repType: "Syntax / Sequential",
      desc: "Deterministic adjacent character transposition in ASCII words (seed=42)",
      fn: (r: DataRecord, rng: () => number) => ({
        text: r.text
          .split(/\s+/)
          .map((w) => {
            if (w.length > 4 && /^[a-zA-Z]+$/.test(w) && rng() < 0.25) {
              const idx = 1 + Math.floor(rng() * (w.length - 3));
              return w.slice(0, idx) + w[idx + 1] + w[idx] + w.slice(idx + 2);
            }
            return w;
          })
          .join(" "),
        context: r.context,
      }),
    },
    {
      name: "Polite Negation Inversion",
      repType: "Syntax / Sequential",
      desc: "Prepend literal politeness hedge ('Honestly, ')",
      fn: (r: DataRecord) => ({ text: `Honestly, ${r.text}`, context: r.context }),
    },
  ];

  const rows = [
    "PerturbationTest,RepresentationType,StabilityPercentage,MeanProbabilityDelta,PredictionFlips,SampleCount,Description",
  ];
  for (let i = 0; i < perturbations.length; i++) {
    const p = perturbations[i];
    const rng = createSeededPRNG(42 + i);
    let flips = 0, sumDelta = 0;
    for (const r of sample) {
      const orig = predictHybrid(r.text, r.context);
      const pert = p.fn(r, rng);
      const res = predictHybrid(pert.text, pert.context);
      if ((orig.p_fused >= 0.5) !== (res.p_fused >= 0.5)) flips++;
      sumDelta += Math.abs(orig.p_fused - res.p_fused);
    }
    const stability = Math.round(((sample.length - flips) / sample.length) * 1000) / 10;
    const meanDelta = Math.round((sumDelta / sample.length) * 1000) / 1000;
    rows.push(`"${p.name}","${p.repType}",${stability},${meanDelta},${flips},${sample.length},"${p.desc}"`);
  }

  fs.mkdirSync(path.resolve("./eval_results/stage5"), { recursive: true });
  fs.writeFileSync(path.resolve("./eval_results/stage5/robustness_stage5.csv"), rows.join("\n") + "\n");
  console.log("Stage 5 deterministic validation robustness CSV saved.");
}

const isMain =
  typeof process !== "undefined" &&
  process.argv[1] &&
  (process.argv[1].endsWith("run_stage5_robustness.ts") ||
    process.argv[1].endsWith("run_stage5_robustness.js"));

if (isMain) {
  runRobustnessEvaluation().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
