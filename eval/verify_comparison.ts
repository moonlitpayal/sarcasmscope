import * as fs from "fs";
import * as path from "path";
import { BENCHMARK_CORPUS } from "./benchmarkCorpus";
import { UNSEEN_TEST_CASES } from "./test_unseen";
import {
  predict as predictV2Linguistic,
  tokenizeWords,
  emojiFeatures,
  cleanText,
} from "../src/engine/sarcasmEngine";
import { predictStatistical } from "../src/engine/statisticalModel";
import { predictHybrid, calculateECE } from "../src/engine/hybridFusion";

// =========================================================================
// PREVIOUS IMPLEMENTATION LOGIC (Pre-recall fix baseline evaluated Nov 7)
// =========================================================================
const PREV_POSITIVE_CUES = new Set([
  "brilliant", "genius", "amazing", "wonderful", "outstanding", "perfect",
  "love", "lovely", "blessed", "greatest", "fantastic", "superb", "flawless", "hero",
  "great", "awesome", "excellent", "magnificent", "masterpiece", "blessing", "pleasure",
  "बढ़िया", "शानदार", "कमाल", "लाजवाब", "वाह", "गजब", "महान", "शाबाश", "सुंदर", "अद्भुत", "धन्य", "बधाई",
  "badiya", "bohot badiya", "gazab", "shandar", "kamaal", "lajawab", "waah", "wah", "zabardast", "sahi", "badhiya",
  "भारी", "छान", "उत्कृष्ट", "मस्त", "शाब्बास", "कौतुक", "bhari", "lai bhari", "mast", "chhan", "lai", "changla",
]);

const PREV_NEGATIVE_CUES = new Set([
  "terrible", "horrible", "disaster", "fail", "failed", "worst", "idiot",
  "useless", "garbage", "trash", "broken", "nightmare", "clown", "stupid", "pathetic", "ruined",
  "delay", "delayed", "late", "traffic", "stuck", "crash", "crashed", "cancel", "canceled", "mess",
  "बेकार", "घटिया", "तमाशा", "पागल", "फालतू", "झूठ", "धोखा", "गंदा", "कचरा", "बकवास", "मूर्ख", "परेशान",
  "bekar", "ghatiya", "tamasha", "pagal", "faltu", "jhoot", "dhoka", "ganda", "bakwas", "kachra", "fas", "late",
  "भिकार", "घाणेरडा", "त्रास", "खड्डे", "खोळंबा", "bhikaar", "tras", "murkh", "khadde", "fasle", "shiksha",
]);

const PREV_INTENSIFIERS = new Set([
  "obviously", "clearly", "totally", "absolutely", "surely", "of course", "definitely",
  "truly", "purely", "literally",
  "ekdam", "khoop", "agdi", "nakkich", "bilkul", "sach me", "kya baat", "bade log", "arre waah", "bohot", "bahut", "khup", "lai",
]);

const PREV_CONTRASTS = new Set([
  "but", "yet", "however", "although", "though", "nevertheless",
  "pan", "pun", "par", "lekin", "magar", "kintu", "पण", "परंतु", "लेकिन", "पर", "मगर",
]);

function prevExtractIncongruity(text: string, context = "") {
  const tokens = tokenizeWords(text).map((t) => t.toLowerCase());
  const detectedPos: string[] = [];
  const detectedNeg: string[] = [];
  const detectedInten: string[] = [];
  let contrastCount = 0;

  for (const t of tokens) {
    const stripped = t.replace(/^[^\w\u0900-\u097f]+|[^\w\u0900-\u097f]+$/gu, "");
    if (PREV_POSITIVE_CUES.has(stripped)) detectedPos.push(t);
    if (PREV_NEGATIVE_CUES.has(stripped)) detectedNeg.push(t);
    if (PREV_INTENSIFIERS.has(stripped)) detectedInten.push(t);
    if (PREV_CONTRASTS.has(stripped)) contrastCount++;
  }

  const posCount = detectedPos.length;
  const negCount = detectedNeg.length;
  const intenCount = detectedInten.length;
  const clash = posCount > 0 && negCount > 0 ? 1.0 : 0.0;

  let ctxClash = 0.0;
  if (context.trim()) {
    const ctxTokens = tokenizeWords(context).map((t) =>
      t.toLowerCase().replace(/^[^\w\u0900-\u097f]+|[^\w\u0900-\u097f]+$/gu, "")
    );
    const ctxNeg = ctxTokens.filter((t) => PREV_NEGATIVE_CUES.has(t)).length;
    if (ctxNeg > 0 && posCount > 0) ctxClash = 1.0;
  }

  const idx = clash * 0.4 + ctxClash * 0.35 + Math.min(contrastCount, 2) * 0.15 + Math.min(intenCount, 2) * 0.1;
  return { posCount, negCount, intenCount, contrastCount, clash, ctxClash, idx };
}

export function prevPredict(text: string, context = "") {
  const clean = cleanText(text);
  const feats = prevExtractIncongruity(clean, context);
  const em = emojiFeatures(text);

  let score = 0.22;
  score += feats.idx * 0.58;
  if (feats.clash > 0) score += 0.25;
  if (feats.ctxClash > 0) score += 0.28;
  if (em.n_irony_emoji > 0) score += Math.min(em.n_irony_emoji * 0.26, 0.45);
  if (feats.posCount > 0 && feats.negCount === 0 && feats.ctxClash === 0) score -= 0.22;
  if (feats.posCount === 0 && feats.negCount === 0 && em.n_irony_emoji === 0 && feats.intenCount === 0) score -= 0.12;
  if (/(\.\.\.|!\?|\?!|!!)/.test(text)) score += 0.08;

  const p = Math.max(0.02, Math.min(0.98, score));
  return { pSarcastic: Math.round(p * 1000) / 1000, isSarcastic: p >= 0.5 };
}

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}

export function computeComparisonMetrics(probs: number[], yTrue: number[]) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const yHat = probs[i] >= 0.5 ? 1 : 0;
    if (yTrue[i] === 1 && yHat === 1) tp++;
    else if (yTrue[i] === 0 && yHat === 1) fp++;
    else if (yTrue[i] === 0 && yHat === 0) tn++;
    else fn++;
  }

  const n = yTrue.length;
  const accuracy = ((tp + tn) / n) * 100;
  const precision = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
  const recall = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const specificity = tn + fp > 0 ? (tn / (tn + fp)) * 100 : 0;

  const prec0 = tn + fn > 0 ? (tn / (tn + fn)) * 100 : 0;
  const rec0 = specificity;
  const f10 = prec0 + rec0 > 0 ? (2 * prec0 * rec0) / (prec0 + rec0) : 0;
  const macroF1 = (f1 + f10) / 2;
  const cal = calculateECE(probs, yTrue);

  return {
    n, tp, fp, tn, fn,
    accuracy: round2(accuracy),
    precision: round2(precision),
    recall: round2(recall),
    f1: round2(f1),
    macroF1: round2(macroF1),
    specificity: round2(specificity),
    ece: cal.ece,
    brier: cal.brier_score,
    nll: cal.nll,
  };
}

function runSuiteOnItems(
  suiteName: string,
  items: { id: string; text: string; context: string; label: number }[]
) {
  const yTrue = items.map((x) => x.label);
  const pV1 = items.map((x) => prevPredict(x.text, x.context).pSarcastic);
  const pV2 = items.map((x) => predictV2Linguistic(x.text, x.context).p_sarcastic);
  const pStat = items.map((x) => predictStatistical(x.text, x.context).p_statistical);
  const pHyb = items.map((x) => predictHybrid(x.text, x.context).p_fused);

  const mV1 = computeComparisonMetrics(pV1, yTrue);
  const mV2 = computeComparisonMetrics(pV2, yTrue);
  const mStat = computeComparisonMetrics(pStat, yTrue);
  const mHyb = computeComparisonMetrics(pHyb, yTrue);

  console.log(`\n--- ${suiteName} (N = ${items.length}) ---`);
  console.log("System                          | Acc    | Prec   | Rec    | MF1    | Spec   | TP/FP/TN/FN");
  console.log("--------------------------------+--------+--------+--------+--------+--------+----------------");
  for (const [name, m] of [
    ["v1 Prev Symbolic (Nov 7)", mV1],
    ["v2 Symbolic Linguistic", mV2],
    ["Level-1 Statistical Ensemble", mStat],
    ["SARcasmScope Calibrated Hybrid", mHyb],
  ] as const) {
    console.log(
      `${name.padEnd(31)} | ${`${m.accuracy}%`.padEnd(6)} | ${`${m.precision}%`.padEnd(6)} | ${`${m.recall}%`.padEnd(6)} | ${`${m.macroF1}%`.padEnd(6)} | ${`${m.specificity}%`.padEnd(6)} | ${m.tp}/${m.fp}/${m.tn}/${m.fn}`
    );
  }

  return { n: items.length, v1Prev: mV1, v2Linguistic: mV2, statisticalOOF: mStat, calibratedHybrid: mHyb };
}

async function main() {
  console.log("================================================================================");
  console.log("SARcasmScope: Reproducible 4-Way Code-Version Comparison Across All Benchmarks");
  console.log("================================================================================");

  const bench30 = runSuiteOnItems(
    "Benchmark Corpus (Closed-Loop Diagnostic)",
    BENCHMARK_CORPUS.map((b) => ({ id: b.id, text: b.text, context: b.context, label: b.groundTruth }))
  );

  const unseen15 = runSuiteOnItems(
    "Unseen Test Cases (Internal Regression)",
    UNSEEN_TEST_CASES.map((u) => ({ id: u.id, text: u.text, context: u.context, label: u.expected }))
  );

  const testPath = path.resolve("./eval_results/independent/test.jsonl");
  const testRecords = fs
    .readFileSync(testPath, "utf8")
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l));

  const heldOut1123 = runSuiteOnItems(
    "Frozen Held-Out Test Split (test.jsonl)",
    testRecords.map((r: any) => ({
      id: r.id,
      text: r.text,
      context: r.context,
      label: r.binary_sarcasm_label,
    }))
  );

  const reportPath = path.resolve("./eval_results/independent/version_comparison_report.json");
  fs.writeFileSync(
    reportPath,
    JSON.stringify({ timestamp: new Date().toISOString(), bench30, unseen15, heldOut1123 }, null, 2) + "\n"
  );
  console.log(`\nSaved version comparison report to ${reportPath}\n`);
}

const isMain =
  typeof process !== "undefined" &&
  process.argv[1] &&
  (process.argv[1].endsWith("verify_comparison.ts") ||
    process.argv[1].endsWith("verify_comparison.js"));

if (isMain) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
