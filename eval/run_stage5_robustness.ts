/**
 * SARcasmScope — Stage 5E: Robustness, Perturbation & Order Sensitivity Evaluation
 * Evaluated on Validation Partition
 */

import * as fs from "fs";
import * as path from "path";
import { predict as predictLinguistic, cleanText } from "../src/engine/sarcasmEngine";
import { predictHybrid, DEFAULT_FUSION_WEIGHTS } from "../src/engine/hybridFusion";

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

// Subwords (length 3-6)
function getSubwords(tokens: string[]): string[] {
  const sw: string[] = [];
  for (const t of tokens) {
    if (t.length <= 4) sw.push(t);
    else {
      for (let i = 0; i <= t.length - 3; i++) {
        sw.push(t.substring(i, Math.min(t.length, i + 3)));
        if (i + 4 <= t.length) sw.push(t.substring(i, i + 4));
      }
    }
  }
  return sw;
}

class FastClassifier {
  vocab: Map<string, number> = new Map();
  idf: Float64Array = new Float64Array(0);
  weights: Float64Array = new Float64Array(0);
  bias = 0.0;

  fit(records: DataRecord[]) {
    const docFreq: Map<string, number> = new Map();
    const tokenized = records.map((r) => getSubwords(getWords(`${r.context} ${r.text}`)));
    for (const tokens of tokenized) {
      for (const t of new Set(tokens)) {
        docFreq.set(t, (docFreq.get(t) || 0) + 1);
      }
    }
    const candidates = Array.from(docFreq.entries())
      .filter(([_, df]) => df >= 3)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 7500);

    candidates.forEach(([term], idx) => this.vocab.set(term, idx));
    const N = records.length;
    const V = this.vocab.size;
    this.idf = new Float64Array(V);
    this.weights = new Float64Array(V);
    candidates.forEach(([term, df], idx) => {
      this.idf[idx] = Math.log((N + 1) / (df + 1)) + 1.0;
    });

    const lr = 0.02;
    const lambda = 1e-3;
    for (let ep = 0; ep < 5; ep++) {
      for (let i = 0; i < records.length; i++) {
        const y = records[i].binary_sarcasm_label;
        const tokens = tokenized[i];
        let dot = this.bias;
        const tfMap: Map<number, number> = new Map();
        for (const t of tokens) {
          const idx = this.vocab.get(t);
          if (idx !== undefined) tfMap.set(idx, (tfMap.get(idx) || 0) + 1);
        }
        for (const [idx, tf] of tfMap.entries()) dot += this.weights[idx] * (tf * this.idf[idx]);
        dot = Math.max(-2.5, Math.min(2.5, dot));
        const p = 1.0 / (1.0 + Math.exp(-dot));
        const grad = p - y;
        for (const [idx, tf] of tfMap.entries()) {
          this.weights[idx] -= lr * (grad * tf * this.idf[idx] + lambda * this.weights[idx]);
        }
        this.bias -= lr * grad;
      }
    }
  }

  predict(tokens: string[]): number {
    let dot = this.bias;
    const tfMap: Map<number, number> = new Map();
    for (const t of tokens) {
      const idx = this.vocab.get(t);
      if (idx !== undefined) tfMap.set(idx, (tfMap.get(idx) || 0) + 1);
    }
    for (const [idx, tf] of tfMap.entries()) dot += this.weights[idx] * (tf * this.idf[idx]);
    dot = Math.max(-2.5, Math.min(2.5, dot));
    return 1.0 / (1.0 + Math.exp(-dot));
  }
}

async function runRobustnessEvaluation() {
  console.log("Running Stage 5E Robustness & Sensitivity Evaluation on Validation Partition...");
  const trainRecords = loadJsonl(path.resolve("./eval_results/independent/train.jsonl"));
  const valRecords = loadJsonl(path.resolve("./eval_results/independent/val.jsonl"));

  const model = new FastClassifier();
  model.fit(trainRecords);

  // Select 100 balanced validation samples (50 sarc / 50 lit)
  const sarcVal = valRecords.filter((r) => r.binary_sarcasm_label === 1).slice(0, 50);
  const litVal = valRecords.filter((r) => r.binary_sarcasm_label === 0).slice(0, 50);
  const testSample = [...sarcVal, ...litVal];

  interface PerturbationTest {
    name: string;
    description: string;
    transform: (r: DataRecord) => { text: string; context: string };
    representationType: "Order-Invariant Bag-of-Tokens" | "Syntax / Sequential";
  }

  const perturbations: PerturbationTest[] = [
    {
      name: "Word Shuffling (Permutation)",
      description: "Randomly shuffle the word order within the target utterance",
      representationType: "Order-Invariant Bag-of-Tokens",
      transform: (r) => {
        const words = r.text.split(/\s+/);
        const shuffled = [...words].sort(() => 0.5 - Math.random());
        return { text: shuffled.join(" "), context: r.context };
      },
    },
    {
      name: "Punctuation Stripping",
      description: "Remove all exclamation marks, question marks, ellipses, and quotes",
      representationType: "Syntax / Sequential",
      transform: (r) => ({
        text: r.text.replace(/[!?.…'"]/g, " "),
        context: r.context.replace(/[!?.…'"]/g, " "),
      }),
    },
    {
      name: "Dialogue Context Removal",
      description: "Detach previous conversational turns, testing ungrounded utterance",
      representationType: "Syntax / Sequential",
      transform: (r) => ({ text: r.text, context: "" }),
    },
    {
      name: "Typographical Noise (Character Swap)",
      description: "Swap adjacent characters in 15% of words to simulate mobile typing errors",
      representationType: "Syntax / Sequential",
      transform: (r) => {
        const words = r.text.split(/\s+/).map((w) => {
          if (w.length > 4 && Math.random() < 0.25) {
            const idx = 1 + Math.floor(Math.random() * (w.length - 3));
            return w.substring(0, idx) + w[idx + 1] + w[idx] + w.substring(idx + 2);
          }
          return w;
        });
        return { text: words.join(" "), context: r.context };
      },
    },
    {
      name: "Polite Negation Inversion",
      description: "Prepend literal politeness hedge ('Honestly, ' or 'Truly, ')",
      representationType: "Syntax / Sequential",
      transform: (r) => ({
        text: `Honestly, ${r.text}`,
        context: r.context,
      }),
    },
  ];

  const results: {
    testName: string;
    repType: string;
    stabilityPct: number;
    meanDeltaProb: number;
    flipCount: number;
    description: string;
  }[] = [];

  for (const p of perturbations) {
    let flips = 0;
    let sumDelta = 0;
    const n = testSample.length;

    for (const r of testSample) {
      const origSem = model.predict(getSubwords(getWords(`${r.context} ${r.text}`)));
      const origRes = predictHybrid(r.text, r.context, { p_transformer: origSem });
      const origPred = origRes.p_fused >= 0.5 ? 1 : 0;

      const perturbed = p.transform(r);
      const pertSem = model.predict(getSubwords(getWords(`${perturbed.context} ${perturbed.text}`)));
      const pertRes = predictHybrid(perturbed.text, perturbed.context, { p_transformer: pertSem });
      const pertPred = pertRes.p_fused >= 0.5 ? 1 : 0;

      if (origPred !== pertPred) flips++;
      sumDelta += Math.abs(origRes.p_fused - pertRes.p_fused);
    }

    const stability = Math.round(((n - flips) / n) * 1000) / 10;
    const meanDelta = Math.round((sumDelta / n) * 1000) / 1000;

    results.push({
      testName: p.name,
      repType: p.representationType,
      stabilityPct: stability,
      meanDeltaProb: meanDelta,
      flipCount: flips,
      description: p.description,
    });
  }

  console.log("=".repeat(110));
  console.log("Perturbation Test".padEnd(35) + "Representation Type".padEnd(32) + "Stability".padEnd(14) + "Mean ΔP".padEnd(12) + "Flips (/100)");
  console.log("-".repeat(110));
  for (const r of results) {
    console.log(
      r.testName.padEnd(35) +
      r.repType.padEnd(32) +
      `${r.stabilityPct}%`.padEnd(14) +
      `${r.meanDeltaProb}`.padEnd(12) +
      `${r.flipCount}`
    );
  }
  console.log("=".repeat(110));

  const csvContent = [
    "PerturbationTest,RepresentationType,StabilityPercentage,MeanProbabilityDelta,PredictionFlips,SampleCount,Description",
    ...results.map(
      (r) =>
        `"${r.testName}","${r.repType}",${r.stabilityPct},${r.meanDeltaProb},${r.flipCount},100,"${r.description}"`
    ),
  ].join("\n") + "\n";

  fs.writeFileSync(path.resolve("./eval_results/stage5/robustness_stage5.csv"), csvContent);
  console.log("Saved robustness report to eval_results/stage5/robustness_stage5.csv");
}

runRobustnessEvaluation();
