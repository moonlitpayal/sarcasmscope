/**
 * Lexicon Mining Pipeline strictly on eval_results/independent/train.jsonl
 * NEVER touches val.jsonl or test.jsonl
 */

import * as fs from "fs";
import * as path from "path";

const TRAIN_PATH = path.resolve("./eval_results/independent/train.jsonl");
const lines = fs.readFileSync(TRAIN_PATH, "utf8").trim().split("\n");

console.log(`Loaded ${lines.length} training records for vocabulary mining.`);

// Basic stopword list for English and Indic to avoid mining non-informative frequent tokens
const STOPWORDS = new Set([
  "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "is", "are", "was", "were",
  "it", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "my", "your",
  "his", "her", "our", "their", "me", "him", "us", "them", "be", "been", "being", "have", "has",
  "had", "do", "does", "did", "with", "as", "by", "from", "about", "into", "through", "after",
  "ka", "ki", "ke", "ko", "se", "me", "mein", "par", "hai", "hain", "tha", "thi", "the", "ho",
  "का", "की", "के", "को", "से", "में", "पर", "है", "हैं", "था", "थी", "थे", "हो", "आणि", "या", "ने",
  "మరియు", "యొక్క", "ఈ", "ఆ", "ఒక"
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/@\w+/g, " ")
    .replace(/[\p{P}\p{S}]+/gu, " ")
    .split(/\s+/)
    .filter((w) => w.length >= 3 && !STOPWORDS.has(w));
}

const sarcWordFreq = new Map<string, number>();
const litWordFreq = new Map<string, number>();
const wordByLang = new Map<string, string>();

for (const line of lines) {
  const rec = JSON.parse(line);
  const words = tokenize(rec.text);
  const isSarc = rec.binary_sarcasm_label === 1;

  for (const w of words) {
    wordByLang.set(w, rec.language);
    if (isSarc) {
      sarcWordFreq.set(w, (sarcWordFreq.get(w) || 0) + 1);
    } else {
      litWordFreq.set(w, (litWordFreq.get(w) || 0) + 1);
    }
  }
}

interface TermStats {
  term: string;
  lang: string;
  sarcCount: number;
  litCount: number;
  total: number;
  sarcPrecision: number;
}

const terms: TermStats[] = [];
const allVocab = new Set([...sarcWordFreq.keys(), ...litWordFreq.keys()]);

for (const w of allVocab) {
  const s = sarcWordFreq.get(w) || 0;
  const l = litWordFreq.get(w) || 0;
  const total = s + l;
  if (total >= 4) { // Minimum frequency support >= 4 in training set
    const prec = s / total;
    terms.push({
      term: w,
      lang: wordByLang.get(w) || "unknown",
      sarcCount: s,
      litCount: l,
      total,
      sarcPrecision: prec,
    });
  }
}

// Filter terms with high sarcasm precision (>= 0.75) and high literal precision (<= 0.20)
const minedSarcasticCues = terms
  .filter((t) => t.sarcPrecision >= 0.70 && t.sarcCount >= 3)
  .sort((a, b) => b.sarcPrecision - a.sarcPrecision || b.sarcCount - a.sarcCount);

const minedLiteralCues = terms
  .filter((t) => t.sarcPrecision <= 0.25 && t.litCount >= 4)
  .sort((a, b) => a.sarcPrecision - b.sarcPrecision || b.litCount - a.litCount);

console.log(`\nDiscovered ${minedSarcasticCues.length} high-confidence sarcasm-associated cues (precision >= 70%, support >= 3).`);
console.log(`Discovered ${minedLiteralCues.length} high-confidence literal-associated cues (sarcasm precision <= 25%, support >= 4).`);

console.log("\nSample top sarcastic cues from train.jsonl:");
minedSarcasticCues.slice(0, 20).forEach((t) => {
  console.log(`- "${t.term}" (${t.lang}): Sarc=${t.sarcCount}, Lit=${t.litCount}, Prec=${(t.sarcPrecision * 100).toFixed(1)}%`);
});

console.log("\nSample top literal cues from train.jsonl:");
minedLiteralCues.slice(0, 20).forEach((t) => {
  console.log(`- "${t.term}" (${t.lang}): Sarc=${t.sarcCount}, Lit=${t.litCount}, Prec=${(t.sarcPrecision * 100).toFixed(1)}%`);
});

// Write mined lexicons to JSON
fs.writeFileSync(
  path.resolve("./eval_results/independent/mined_training_lexicons.json"),
  JSON.stringify(
    {
      provenance: "Strictly mined from eval_results/independent/train.jsonl (Seed 42)",
      training_samples_analyzed: lines.length,
      mined_sarcastic_cues: minedSarcasticCues,
      mined_literal_cues: minedLiteralCues,
    },
    null,
    2
  )
);
console.log("\nSaved mined lexicons to eval_results/independent/mined_training_lexicons.json");
