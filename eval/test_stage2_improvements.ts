/**
 * Comprehensive Stage 2 Evaluation Suite
 * 
 * Verifies:
 * 1. Indic Morphological Suffix Stripping (Hindi, Marathi, Telugu)
 * 2. Romanized Phonetic Normalization (Hinglish, Marathi-Latin)
 * 3. Lexical Coverage Density C(x) & Epistemic Uncertainty U(x)
 * 4. False-Confidence Correction & Abstention behavior
 * 5. Deadpan & Counterfactual Sarcasm Detection
 * 6. Sincere Praise & Direct Complaint Specificity (No False Positives)
 * 7. Empirical Evaluation on independent VALIDATION Split (val.jsonl)
 *    [NOTE: Held-out test.jsonl is STRICTLY UNTOUCHED]
 */

import {
  predict,
  stemIndicToken,
  normalizeRomanized,
  extractIncongruity,
  POSITIVE_CUES,
  NEGATIVE_CUES,
} from "../src/engine/sarcasmEngine";
import * as fs from "fs";
import * as path from "path";

console.log("================================================================================");
console.log("SARcasmScope: Stage 2 Client-Side Linguistic Engine Comprehensive Verification");
console.log("================================================================================\n");

// -----------------------------------------------------------------------------
// PART 1: LINGUISTIC UNIT TESTS
// -----------------------------------------------------------------------------
const LINGUISTIC_CASES = [
  // 1. Hindi suffix variations
  {
    category: "Hindi Suffix Variation",
    text: "वाह क्या शानदार बात है, रेलवे की खराबियों पर मंत्री जी को बधाई! 🙄",
    context: "ट्रेन लेट थी",
    expectedVerdict: "Sarcastic",
    notes: "Should stem 'खराबियों' -> 'खराब' (negative cue) matching with 'शानदार'/'बधाई'",
  },
  // 2. Marathi suffix variations
  {
    category: "Marathi Suffix Variation",
    text: "खूप छान नियोजन आहे, पालिकेच्या खड्ड्यांत गाडी आदळल्यामुळे पाठीचे मणके तुटले!",
    context: "खड्ड्यात अपघात",
    expectedVerdict: "Sarcastic",
    notes: "Should stem 'खड्ड्यांत' -> 'खड्ड' (negative) and match with 'खूप छान'",
  },
  // 3. Telugu suffix variations
  {
    category: "Telugu Suffix Variation",
    text: "వావ్ గొప్ప పని, రైలు తీవ్ర ఆలస్యమైనందుకు అందరూ చప్పట్లు కొడుతున్నారు!",
    context: "ట్రైన్ లేట్",
    expectedVerdict: "Sarcastic",
    notes: "Should stem 'ఆలస్యమైనందుకు' -> 'ఆలస్య' (negative) matching 'గొప్ప'",
  },
  // 4. Hinglish phonetic variations
  {
    category: "Hinglish Phonetic Variation",
    text: "Baaahuuut shaandaaar speed yaar, ek page khulne me aadha ghanta lag gaya! 🙃",
    context: "Internet speed drop",
    expectedVerdict: "Sarcastic",
    notes: "Phonetic folding: 'baaahuuut' -> 'bahut', 'shaandaaar' -> 'shandar'",
  },
  // 5. Romanized Marathi variations
  {
    category: "Romanized Marathi Variation",
    text: "Laaaai bhaari नियोजन, don taas traffic madhe thevlya baddal khup aabhar! 👏",
    context: "Traffic jam",
    expectedVerdict: "Sarcastic",
    notes: "Deduplication: 'laaaai' -> 'lai', 'bhaari' -> 'bhari'",
  },
  // 6. Counterfactual sarcasm
  {
    category: "Counterfactual Sarcasm",
    text: "As if the corrupt politicians actually care about the taxpayers. Sure.",
    context: "Tax hike announcement",
    expectedVerdict: "Sarcastic",
    notes: "Discourse marker 'as if' + negative 'corrupt' + 'care'",
  },
  // 7. Direct negative complaint (MUST NOT BE SARCASTIC)
  {
    category: "Direct Complaint (Literal Negative)",
    text: "The airline cancelled our flight without notice and lost all three suitcases.",
    context: "Luggage lost",
    expectedVerdict: "Literal",
    notes: "Direct complaint: has negative cues, zero praise, zero irony emoji -> must stay Literal",
  },
  // 8. Sincere praise (MUST NOT BE SARCASTIC)
  {
    category: "Sincere Praise (Literal Positive)",
    text: "The delivery arrived ahead of time and customer support was very helpful and clear.",
    context: "Good shipping experience",
    expectedVerdict: "Literal",
    notes: "Sincere appreciation: positive cues with zero clash -> must stay Literal",
  },
  // 9. Low-coverage ambiguous input (MUST TRIGGER UNCERTAIN / ABSTAIN)
  {
    category: "Low-Coverage Ambiguous Input",
    text: "Quantum chromodynamics reveals asymptotic freedom in non-abelian gauge theories.",
    context: "Physics lecture",
    expectedVerdict: "Uncertain",
    notes: "High epistemic uncertainty and low domain coverage -> should abstain!",
  },
];

console.log("--- PART 1: LINGUISTIC TARGETED TEST CASES ---");
let part1Passed = 0;
for (const tc of LINGUISTIC_CASES) {
  const res = predict(tc.text, tc.context);
  const match = res.verdict === tc.expectedVerdict;
  if (match) part1Passed++;
  console.log(
    `[${match ? "PASS" : "FAIL"}] ${tc.category.padEnd(35)} | Got: ${res.verdict.padEnd(9)} (P=${res.p_sarcastic.toFixed(3)}, Conf=${res.confidence.toFixed(3)}, Cov=${res.incongruity.lexical_coverage.toFixed(2)}, Uncert=${res.incongruity.epistemic_uncertainty.toFixed(2)}) | Exp: ${tc.expectedVerdict}`
  );
}
console.log(`Part 1 Score: ${part1Passed}/${LINGUISTIC_CASES.length} passed.\n`);

// -----------------------------------------------------------------------------
// PART 2: EMPIRICAL EVALUATION ON VALIDATION SPLIT (val.jsonl)
// -----------------------------------------------------------------------------
console.log("--- PART 2: EMPIRICAL VALIDATION SPLIT EVALUATION (val.jsonl) ---");
console.log("Evaluating Stage 2 engine on 1,104 independent validation samples...");
const valPath = path.resolve("./eval_results/independent/val.jsonl");
const valLines = fs.readFileSync(valPath, "utf8").trim().split("\n");

let tp = 0, fp = 0, tn = 0, fn = 0, abstains = 0;
let valCorrect = 0;
const perDatasetStats: Record<string, { total: number; tp: number; fp: number; tn: number; fn: number; abstain: number }> = {};

for (const line of valLines) {
  const rec = JSON.parse(line);
  const res = predict(rec.text, rec.context || "");
  const isSarc = res.p_sarcastic >= 0.5 ? 1 : 0;
  const trueLabel = rec.binary_sarcasm_label;

  if (!perDatasetStats[rec.dataset]) {
    perDatasetStats[rec.dataset] = { total: 0, tp: 0, fp: 0, tn: 0, fn: 0, abstain: 0 };
  }
  const ds = perDatasetStats[rec.dataset];
  ds.total++;

  if (res.abstain) {
    abstains++;
    ds.abstain++;
  }

  if (isSarc === trueLabel) valCorrect++;

  if (trueLabel === 1 && isSarc === 1) { tp++; ds.tp++; }
  else if (trueLabel === 0 && isSarc === 1) { fp++; ds.fp++; }
  else if (trueLabel === 0 && isSarc === 0) { tn++; ds.tn++; }
  else if (trueLabel === 1 && isSarc === 0) { fn++; ds.fn++; }
}

const totalVal = valLines.length;
const valAcc = (valCorrect / totalVal) * 100;
const valPrec = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
const valRec = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
const valF1 = valPrec + valRec > 0 ? (2 * valPrec * valRec) / (valPrec + valRec) : 0;
const specificity = tn + fp > 0 ? (tn / (tn + fp)) * 100 : 0;

console.log(`\nValidation Performance Summary (N = ${totalVal}):`);
console.log(`- Accuracy:            ${valAcc.toFixed(2)}%`);
console.log(`- Sarcasm Precision:   ${valPrec.toFixed(2)}% (TP=${tp}, FP=${fp})`);
console.log(`- Sarcasm Recall:      ${valRec.toFixed(2)}% (TP=${tp}, FN=${fn})`);
console.log(`- Sarcasm F1 Score:    ${valF1.toFixed(2)}%`);
console.log(`- Specificity:         ${specificity.toFixed(2)}% (TN=${tn}, FP=${fp})`);
console.log(`- Selective Abstention Rate: ${((abstains / totalVal) * 100).toFixed(2)}% (${abstains} ambiguous samples escalated)`);

console.log("\nPer-Dataset Validation Breakdown:");
for (const [dsName, s] of Object.entries(perDatasetStats)) {
  const dPrec = s.tp + s.fp > 0 ? ((s.tp / (s.tp + s.fp)) * 100).toFixed(1) : "0.0";
  const dRec = s.tp + s.fn > 0 ? ((s.tp / (s.tp + s.fn)) * 100).toFixed(1) : "0.0";
  const dSpec = s.tn + s.fp > 0 ? ((s.tn / (s.tn + s.fp)) * 100).toFixed(1) : "0.0";
  console.log(`* ${dsName.padEnd(30)} | N=${s.total.toString().padEnd(4)} | TP=${s.tp} FP=${s.fp} TN=${s.tn} FN=${s.fn} | Prec=${dPrec}% Rec=${dRec}% Spec=${dSpec}% | Abstain=${s.abstain}`);
}

// Write validation metrics CSV
const valCsvHeader = "Dataset,TotalSamples,TP,FP,TN,FN,Precision,Recall,Specificity,AbstainCount\n";
const valCsvRows = Object.entries(perDatasetStats).map(([ds, s]) => {
  const p = s.tp + s.fp > 0 ? (s.tp / (s.tp + s.fp)) * 100 : 0;
  const r = s.tp + s.fn > 0 ? (s.tp / (s.tp + s.fn)) * 100 : 0;
  const sp = s.tn + s.fp > 0 ? (s.tn / (s.tn + s.fp)) * 100 : 0;
  return `"${ds}",${s.total},${s.tp},${s.fp},${s.tn},${s.fn},${p.toFixed(2)},${r.toFixed(2)},${sp.toFixed(2)},${s.abstain}`;
});

fs.writeFileSync(
  path.resolve("./eval_results/independent/stage2_validation_metrics.csv"),
  valCsvHeader + valCsvRows.join("\n") + "\n"
);
console.log("\nSaved validation metrics to eval_results/independent/stage2_validation_metrics.csv");
