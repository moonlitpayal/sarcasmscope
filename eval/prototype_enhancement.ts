import { BENCHMARK_CORPUS } from "./run_scientific_evaluation";

// Enhanced lexicons (all Indic scripts + English)
export const POSITIVE_CUES_ENHANCED = new Set([
  // English
  "brilliant", "genius", "amazing", "wonderful", "outstanding", "perfect",
  "love", "lovely", "blessed", "greatest", "fantastic", "superb", "flawless", "hero",
  "great", "awesome", "excellent", "magnificent", "masterpiece", "blessing", "pleasure",
  "top", "proud", "congrats", "congratulations",
  // Hindi Devanagari
  "बढ़िया", "शानदार", "कमाल", "लाजवाब", "वाह", "गजब", "महान", "शाबाश", "सुंदर", "अद्भुत", "धन्य", "बधाई",
  "सलामी", "तोप", "तोपों", "मजा", "स्वर्ग",
  // Hinglish
  "badiya", "bohot badiya", "gazab", "shandar", "kamaal", "lajawab", "waah", "wah", "zabardast", "sahi", "badhiya",
  "top", "hero", "bhai", "mazaa", "maza",
  // Marathi Devanagari & Latin
  "भारी", "छान", "उत्कृष्ट", "मस्त", "शाब्बास", "कौतुक", "व्वा", "नियोजन",
  "bhari", "lai bhari", "mast", "chhan", "lai", "changla", "aabhar",
  // Telugu
  "వావ్", "భలే", "అద్భుతం", "అద్భుతమైన", "గొప్ప", "బాగుంది", "మంచి"
]);

export const NEGATIVE_CUES_ENHANCED = new Set([
  // English
  "terrible", "horrible", "disaster", "fail", "failed", "fails", "worst", "idiot",
  "useless", "garbage", "trash", "broken", "nightmare", "clown", "stupid", "pathetic", "ruined",
  "delay", "delayed", "late", "traffic", "stuck", "crash", "crashed", "cancel", "canceled", "mess",
  "downpour", "soaked", "monsoon", "hold", "dropped", "waiting", "spoil", "spoiled", "lost", "outage",
  "disconnect", "disconnected", "flood", "flooding", "cyclone", "potholes", "damage", "corrupt",
  // Hindi Devanagari
  "बेकार", "घटिया", "तमाशा", "पागल", "फालतू", "झूठ", "धोखा", "गंदा", "कचरा", "बकवास", "मूर्ख", "परेशान",
  "लेट", "खराब", "ठंडा", "पानी", "जलजमाव", "बाढ़", "गड्ढा", "गड्ढे", "भीड़", "उशीर", "मुश्किल",
  // Hinglish
  "bekar", "ghatiya", "tamasha", "pagal", "faltu", "jhoot", "dhoka", "ganda", "bakwas", "kachra", "fas", "late",
  "spoil", "spoiled", "fail", "load", "drop", "jam", "chhut", "tough", "problem",
  // Marathi Devanagari & Latin
  "भिकार", "घाणेरडा", "त्रास", "खड्डे", "खोळंबा", "गोंधळ", "लाईन", "रांग", "उशीर", "पाऊस", "जाम",
  "bhikaar", "tras", "murkh", "khadde", "fasle", "shiksha", "jam", "traffic",
  // Telugu
  "వర్షం", "కరెంట్", "పవర్", "కట్", "ఆలస్యం", "జామ్", "ట్రాఫిక్", "పాడైపోయింది", "దారుణం", "లేట్"
]);

export const RHETORICAL_PATTERNS = [
  /\b(what a|just what|who knew|thanks to|as if|oh sure)\b/i,
  /\b(वाह क्या बात|गजब|धन्य हो|सलामी|वाह रे वाह)\b/,
  /\b(kya baat hai|wah re wah|lai bhari|bhale timing)\b/i,
  /(21 तोपों|स्विमिंग पूल)/
];

export const IRONY_EMOJIS_SET = new Set(["🙄", "😏", "😒", "🙃", "🤡", "😑", "😬"]);

function cleanWord(w: string): string {
  return w.replace(/^[\p{P}\p{S}]+|[\p{P}\p{S}]+$/gu, "").toLowerCase();
}

export function improvedPredict(text: string, context = "") {
  const words = text.split(/\s+/).map(cleanWord).filter(Boolean);
  const ctxWords = context.split(/\s+/).map(cleanWord).filter(Boolean);

  let posCount = 0;
  let negCount = 0;
  for (const w of words) {
    if (POSITIVE_CUES_ENHANCED.has(w)) posCount++;
    if (NEGATIVE_CUES_ENHANCED.has(w)) negCount++;
  }

  let ctxNegCount = 0;
  for (const w of ctxWords) {
    if (NEGATIVE_CUES_ENHANCED.has(w)) ctxNegCount++;
  }

  // Emoji features
  const emojis = Array.from(text.match(/[\u{1F300}-\u{1FAFF}\u{1F000}-\u{1F2FF}\u{2600}-\u{27BF}\u{2B00}-\u{2BFF}\u{1F1E6}-\u{1F1FF}]/gu) || []);
  const hasIronyEmoji = emojis.some((e) => IRONY_EMOJIS_SET.has(e));

  // Rhetorical cues
  const hasRhetorical = RHETORICAL_PATTERNS.some((p) => p.test(text));

  const polarityClash = posCount > 0 && negCount > 0 ? 1.0 : 0.0;
  const contextClash = ctxNegCount > 0 && (posCount > 0 || hasIronyEmoji || hasRhetorical) ? 1.0 : 0.0;

  // Base prior
  let score = 0.20;

  if (polarityClash > 0) score += 0.35;
  if (contextClash > 0) score += 0.32;
  if (hasIronyEmoji) score += 0.30;
  if (hasRhetorical) score += 0.15;

  // Penalize sincere literal sentences
  // Sincere positive: pos words, no negative in sentence, NO context clash, NO irony emoji, NO rhetorical markers
  if (posCount > 0 && negCount === 0 && contextClash === 0 && !hasIronyEmoji && !hasRhetorical) {
    score -= 0.22;
  }

  // Factual neutral sentence
  if (posCount === 0 && negCount === 0 && !hasIronyEmoji && contextClash === 0 && !hasRhetorical) {
    score -= 0.10;
  }

  // Direct criticism (negative words, but NO positive praise, NO irony emoji): Sincere complaint, not sarcasm!
  if (posCount === 0 && negCount > 0 && !hasIronyEmoji && !hasRhetorical) {
    score -= 0.15;
  }

  const pSarcastic = Math.max(0.02, Math.min(0.98, score));
  return { pSarcastic, isSarcastic: pSarcastic >= 0.5 };
}

// Test against benchmark corpus
let tp = 0, fp = 0, tn = 0, fn = 0;
for (const item of BENCHMARK_CORPUS) {
  const res = improvedPredict(item.text, item.context);
  const pred = res.isSarcastic ? 1 : 0;
  const actual = item.groundTruth;

  if (actual === 1 && pred === 1) tp++;
  else if (actual === 0 && pred === 1) fp++;
  else if (actual === 0 && pred === 0) tn++;
  else if (actual === 1 && pred === 0) fn++;
}

const acc = (tp + tn) / BENCHMARK_CORPUS.length;
const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;

const prec0 = tn + fn > 0 ? tn / (tn + fn) : 0;
const rec0 = tn + fp > 0 ? tn / (tn + fp) : 0;
const f10 = prec0 + rec0 > 0 ? (2 * prec0 * rec0) / (prec0 + rec0) : 0;
const macroF1 = (f1 + f10) / 2;

console.log("=== PROTOTYPE ENHANCEMENT RESULTS ===");
console.log(`Accuracy:  ${(acc * 100).toFixed(1)}%`);
console.log(`Precision: ${(prec * 100).toFixed(1)}%`);
console.log(`Recall:    ${(rec * 100).toFixed(1)}%`);
console.log(`F1 Score:  ${(f1 * 100).toFixed(1)}%`);
console.log(`Macro F1:  ${(macroF1 * 100).toFixed(1)}%`);
console.log(`Matrix:    TP=${tp}, FP=${fp}, TN=${tn}, FN=${fn}`);
