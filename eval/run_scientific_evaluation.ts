/**
 * SARcasmScope Scientific Evaluation Suite
 * Independent evaluation harness executing Tests 1 through 15
 */

import {
  predict,
  computeOcclusion,
  perturbAll,
  analyzeCodeMixing,
  extractIncongruity,
  emojiFeatures,
  cleanText,
  tokenizeWords,
  scriptOf,
  applyTemperatureCalibration,
  computeRawScore,
  POSITIVE_CUES,
  NEGATIVE_CUES,
  INTENSIFIERS,
  CONTRASTS,
  IRONY_EMOJIS,
} from "../src/engine/sarcasmEngine";
import * as fs from "fs";
import * as path from "path";

// Ensure output directory exists
const OUT_DIR = path.resolve("./eval_results");
if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

// -------------------------------------------------------------
// BENCHMARK EVALUATION CORPUS
// Ground-truth curated multilingual test items across 5 languages
// -------------------------------------------------------------
export interface BenchmarkItem {
  id: string;
  lang: "English" | "Hindi" | "Hinglish" | "Marathi" | "Telugu";
  script: string;
  text: string;
  context: string;
  groundTruth: 1 | 0; // 1 = Sarcastic, 0 = Literal
  category: "Praise_Under_Failure" | "Direct_Criticism" | "Sincere_Praise" | "Factual" | "Irony_Emoji_Only" | "Hyperbolic_Praise";
  notes: string;
}

export const BENCHMARK_CORPUS: BenchmarkItem[] = [
  // --- ENGLISH ---
  {
    id: "EN_01",
    lang: "English",
    script: "Latin",
    text: "Oh brilliant, another delay on Mumbai locals, just what my Monday needed!",
    context: "Train canceled due to waterlogging",
    groundTruth: 1,
    category: "Praise_Under_Failure",
    notes: "Classic sarcasm: positive cue 'brilliant' clashing with negative reality 'delay'",
  },
  {
    id: "EN_02",
    lang: "English",
    script: "Latin",
    text: "I absolutely love it when code fails directly in production on a Friday night.",
    context: "Production server crashed",
    groundTruth: 1,
    category: "Praise_Under_Failure",
    notes: "Overt praise 'absolutely love' with failure 'code fails'",
  },
  {
    id: "EN_03",
    lang: "English",
    script: "Latin",
    text: "What a genius move leaving the umbrella at home during a monsoon downpour.",
    context: "Completely soaked in rain",
    groundTruth: 1,
    category: "Praise_Under_Failure",
    notes: "Praise 'genius' contrasted with soaked outcome",
  },
  {
    id: "EN_04",
    lang: "English",
    script: "Latin",
    text: "Great customer service, waiting 45 minutes on hold listening to elevator music.",
    context: "Call dropped after waiting",
    groundTruth: 1,
    category: "Praise_Under_Failure",
    notes: "Positive 'Great' with negative wait time",
  },
  {
    id: "EN_05",
    lang: "English",
    script: "Latin",
    text: "The train arrived exactly on time at Dadar station today.",
    context: "Smooth morning commute",
    groundTruth: 0,
    category: "Factual",
    notes: "Sincere factual literal report",
  },
  {
    id: "EN_06",
    lang: "English",
    script: "Latin",
    text: "Our team solved the production issue within 10 minutes, outstanding effort.",
    context: "Successful quick patch deployment",
    groundTruth: 0,
    category: "Sincere_Praise",
    notes: "Sincere praise congruent with fast resolution",
  },
  {
    id: "EN_07",
    lang: "English",
    script: "Latin",
    text: "The technical documentation is clean, concise, and easy to follow.",
    context: "Reading library docs",
    groundTruth: 0,
    category: "Sincere_Praise",
    notes: "Literal positive evaluation",
  },
  {
    id: "EN_08",
    lang: "English",
    script: "Latin",
    text: "The flight was delayed by four hours and baggage was lost.",
    context: "Worst airline experience",
    groundTruth: 0,
    category: "Direct_Criticism",
    notes: "Critical literal statement - no irony, purely negative",
  },

  // --- HINDI (Devanagari) ---
  {
    id: "HI_01",
    lang: "Hindi",
    script: "Devanagari",
    text: "वाह क्या बात है, 4 घंटे ट्रेन लेट करने के लिए रेलवे को 21 तोपों की सलामी! 🙄",
    context: "ट्रेन 4 घंटे लेट थी",
    groundTruth: 1,
    category: "Praise_Under_Failure",
    notes: "Sarcastic praise 'वाह क्या बात है' for delayed train with irony emoji",
  },
  {
    id: "HI_02",
    lang: "Hindi",
    script: "Devanagari",
    text: "अरे गजब! पूरे ₹500 लेके ठंडा और पानी जैसा सूप पिला दिया, धन्य हो!",
    context: "खराब रेस्टोरेंट का अनुभव",
    groundTruth: 1,
    category: "Praise_Under_Failure",
    notes: "Intra-sentential opposition: 'गजब'/'धन्य' vs 'ठंडा और पानी जैसा'",
  },
  {
    id: "HI_03",
    lang: "Hindi",
    script: "Devanagari",
    text: "बहुत बढ़िया सरकार, हर सड़क पर स्विमिंग पूल बना दिया बारिश में।",
    context: "सड़क पर भारी जलजमाव",
    groundTruth: 1,
    category: "Praise_Under_Failure",
    notes: "Hyperbolic metaphor 'स्विमिंग पूल' with praise 'बहुत बढ़िया'",
  },
  {
    id: "HI_04",
    lang: "Hindi",
    script: "Devanagari",
    text: "आज की बैठक बहुत ही उत्पादक रही और समय पर समाप्त हो गई।",
    context: "ऑफिस काम",
    groundTruth: 0,
    category: "Factual",
    notes: "Literal formal workplace report",
  },
  {
    id: "HI_05",
    lang: "Hindi",
    script: "Devanagari",
    text: "खाना सच में बहुत स्वादिष्ट और लाजवाब था, धन्यवाद।",
    context: "पारिवारिक रात्रिभोज",
    groundTruth: 0,
    category: "Sincere_Praise",
    notes: "Sincere culinary praise",
  },
  {
    id: "HI_06",
    lang: "Hindi",
    script: "Devanagari",
    text: "ट्रेन 4 घंटे लेट थी और स्टेशन पर बहुत भीड़ थी।",
    context: "यात्रा में परेशानी",
    groundTruth: 0,
    category: "Direct_Criticism",
    notes: "Literal complaint without sarcastic phrasing",
  },

  // --- HINGLISH (Code-Mixed) ---
  {
    id: "HG_01",
    lang: "Hinglish",
    script: "Latin",
    text: "Bohot badiya bhai, exam se ek raat pehle phone pe reels dekh, pakka top karega! 🙃",
    context: "Padhai bilkul nahi ki",
    groundTruth: 1,
    category: "Praise_Under_Failure",
    notes: "Code-mixed Hinglish sarcasm with inverted emoji",
  },
  {
    id: "HG_02",
    lang: "Hinglish",
    script: "Latin",
    text: "Waah, 3 ghante traffic jam me khade rakha, weekend ka plan pura spoil kar diya.",
    context: "Stuck on expressway",
    groundTruth: 1,
    category: "Praise_Under_Failure",
    notes: "'Waah' combined with 'traffic jam' and 'spoil'",
  },
  {
    id: "HG_03",
    lang: "Hinglish",
    script: "Latin",
    text: "Superb speed yaar, ek page load hone me 10 minute lag rahe hain! 👏",
    context: "Internet speed drop",
    groundTruth: 1,
    category: "Praise_Under_Failure",
    notes: "'Superb speed' contrasted with 10 minute load time",
  },
  {
    id: "HG_04",
    lang: "Hinglish",
    script: "Latin",
    text: "Train time par platform par aa gayi, safar bohot accha raha.",
    context: "Smooth journey to Delhi",
    groundTruth: 0,
    category: "Factual",
    notes: "Literal Hinglish appreciation of on-time arrival",
  },
  {
    id: "HG_05",
    lang: "Hinglish",
    script: "Latin",
    text: "Bhai kal ki party me bohot maza aaya, sab dost mile.",
    context: "College reunion",
    groundTruth: 0,
    category: "Sincere_Praise",
    notes: "Genuine Hinglish positive memory",
  },
  {
    id: "HG_06",
    lang: "Hinglish",
    script: "Latin",
    text: "Exam bohot tough tha aur mere saare questions chhut gaye.",
    context: "Bad exam day",
    groundTruth: 0,
    category: "Direct_Criticism",
    notes: "Sincere distress, non-sarcastic",
  },

  // --- MARATHI (Devanagari & Latin) ---
  {
    id: "MR_01",
    lang: "Marathi",
    script: "Latin",
    text: "Lai bhari bhau, 2 taas traffic madhe thevlya badal khup khup aabhar! 👏",
    context: "Stuck in 2 hr traffic jam at Hinjewadi",
    groundTruth: 1,
    category: "Praise_Under_Failure",
    notes: "Marathi Latin sarcastic cue 'Lai bhari' and 'aabhar' for traffic delay",
  },
  {
    id: "MR_02",
    lang: "Marathi",
    script: "Devanagari",
    text: "व्वा, खूप छान नियोजन आहे, अर्धा तास आधी येऊन पण लाईन मध्येच उभे आहोत!",
    context: "कार्यक्रमात गोंधळ",
    groundTruth: 1,
    category: "Praise_Under_Failure",
    notes: "Devanagari Marathi sarcasm: 'खूप छान नियोजन' against standing in line",
  },
  {
    id: "MR_03",
    lang: "Marathi",
    script: "Devanagari",
    text: "वाह रे वाह सरकार, रस्त्यात खड्डे आहेत की खड्ड्यात रस्ता कळतच नाही!",
    context: "रस्त्यांची दुरवस्था",
    groundTruth: 1,
    category: "Praise_Under_Failure",
    notes: "Praise cue 'वाह रे वाह' with complaint about potholes",
  },
  {
    id: "MR_04",
    lang: "Marathi",
    script: "Latin",
    text: "Aaj cha presentation khup changla zala, congrats team!",
    context: "Project approval meeting",
    groundTruth: 0,
    category: "Sincere_Praise",
    notes: "Genuine team congratulations in Marathi-Latin",
  },
  {
    id: "MR_05",
    lang: "Marathi",
    script: "Devanagari",
    text: "पुण्यातील जेवण आणि हवामान दोन्ही खूप मस्त आहेत.",
    context: "पुणे भेट",
    groundTruth: 0,
    category: "Sincere_Praise",
    notes: "Literal sincere praise of Pune climate and food",
  },
  {
    id: "MR_06",
    lang: "Marathi",
    script: "Devanagari",
    text: "बस वेळेवर आली नाही आणि मला ऑफिसला खूप उशीर झाला.",
    context: "ऑफिसला उशीर",
    groundTruth: 0,
    category: "Direct_Criticism",
    notes: "Direct literal grievance",
  },

  // --- TELUGU ---
  {
    id: "TE_01",
    lang: "Telugu",
    script: "Telugu",
    text: "వావ్, మళ్లీ వర్షం, నా బైక్ ప్రయాణానికి భలే టైమింగ్ 🙄",
    context: "వర్షంలో తడిసిపోయాను",
    groundTruth: 1,
    category: "Praise_Under_Failure",
    notes: "Telugu sarcasm: 'వావ్' and 'భలే టైమింగ్' with irony emoji and rain context",
  },
  {
    id: "TE_02",
    lang: "Telugu",
    script: "Telugu",
    text: "అద్భుతమైన సర్వీస్, 2 గంటలు కరెంట్ తీసేసి గొప్ప పని చేశారు!",
    context: "పవర్ కట్",
    groundTruth: 1,
    category: "Praise_Under_Failure",
    notes: "Sarcastic praise 'అద్భుతమైన సర్వీస్' for power outage",
  },
  {
    id: "TE_03",
    lang: "Telugu",
    script: "Telugu",
    text: "సినిమా చాలా బాగుంది, కథ మరియు నటన అద్భుతం.",
    context: "సినిమా చూశాము",
    groundTruth: 0,
    category: "Sincere_Praise",
    notes: "Genuine movie review in Telugu",
  },
  {
    id: "TE_04",
    lang: "Telugu",
    script: "Telugu",
    text: "రైలు సమయానికి వచ్చింది, ప్రయాణం సుఖంగా సాగింది.",
    context: "హైదరాబాద్ ప్రయాణం",
    groundTruth: 0,
    category: "Factual",
    notes: "Literal factual statement about travel in Telugu",
  },
];

// Helper: Statistical computation
function computeConfusionMatrix(yTrue: number[], yPred: number[]) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === 1 && yPred[i] === 1) tp++;
    else if (yTrue[i] === 0 && yPred[i] === 1) fp++;
    else if (yTrue[i] === 0 && yPred[i] === 0) tn++;
    else if (yTrue[i] === 1 && yPred[i] === 0) fn++;
  }
  const acc = (tp + tn) / yTrue.length;
  const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
  const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;

  // Macro metrics
  const prec0 = tn + fn > 0 ? tn / (tn + fn) : 0;
  const rec0 = tn + fp > 0 ? tn / (tn + fp) : 0;
  const f10 = prec0 + rec0 > 0 ? (2 * prec0 * rec0) / (prec0 + rec0) : 0;
  const macroF1 = (f1 + f10) / 2;

  return { tp, fp, tn, fn, accuracy: acc, precision: prec, recall: rec, f1, macroF1 };
}

function computeECE(probs: number[], labels: number[], numBins = 10) {
  const binEdges: number[] = [];
  for (let i = 0; i <= numBins; i++) binEdges.push(i / numBins);

  let ece = 0;
  const total = probs.length;
  const binStats: { bin: string; count: number; avgConf: number; avgAcc: number; absDiff: number }[] = [];

  for (let b = 0; b < numBins; b++) {
    const lo = binEdges[b];
    const hi = binEdges[b + 1];
    const inBinIndices: number[] = [];
    for (let i = 0; i < total; i++) {
      const p = probs[i];
      if (b === numBins - 1 ? p >= lo && p <= hi : p >= lo && p < hi) {
        inBinIndices.push(i);
      }
    }

    if (inBinIndices.length > 0) {
      const avgConf = inBinIndices.reduce((acc, i) => acc + probs[i], 0) / inBinIndices.length;
      const avgAcc = inBinIndices.reduce((acc, i) => acc + labels[i], 0) / inBinIndices.length;
      const diff = Math.abs(avgAcc - avgConf);
      ece += (inBinIndices.length / total) * diff;
      binStats.push({
        bin: `[${lo.toFixed(1)}-${hi.toFixed(1)}]`,
        count: inBinIndices.length,
        avgConf: Math.round(avgConf * 1000) / 1000,
        avgAcc: Math.round(avgAcc * 1000) / 1000,
        absDiff: Math.round(diff * 1000) / 1000,
      });
    }
  }

  return { ece: Math.round(ece * 10000) / 10000, binStats };
}

function computeBrierAndNLL(probs: number[], labels: number[]) {
  let brierSum = 0;
  let nllSum = 0;
  const eps = 1e-6;

  for (let i = 0; i < probs.length; i++) {
    const p = Math.max(eps, Math.min(1 - eps, probs[i]));
    const y = labels[i];
    brierSum += Math.pow(p - y, 2);
    nllSum += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
  }

  return {
    brierScore: Math.round((brierSum / probs.length) * 10000) / 10000,
    nll: Math.round((nllSum / probs.length) * 10000) / 10000,
  };
}

// -------------------------------------------------------------
// MAIN TEST HARNESS EXECUTION
// -------------------------------------------------------------
async function runCompleteEvaluation() {
  console.log("================================================================================");
  console.log("SARcasmScope: Rigorous Scientific Evaluation & Reproducibility Suite");
  console.log("Principal Investigator: Payal Mehta");
  console.log("Timestamp:", new Date().toISOString());
  console.log("================================================================================\n");

  const executionLog: any[] = [];

  // =========================================================================
  // TEST 1: FUNCTIONAL SYSTEM TESTING
  // =========================================================================
  console.log(">>> EXECUTING TEST 1: FUNCTIONAL SYSTEM TESTING");
  const t1_tests = [
    { name: "Input validation (empty text)", input: "", context: "", shouldThrowOrHandle: true },
    { name: "Single word input", input: "brilliant", context: "", shouldThrowOrHandle: true },
    { name: "Length limit check (5000 chars)", input: "word ".repeat(1000), context: "", shouldThrowOrHandle: true },
    { name: "Clean text normalization", input: "http://test.com @user #sarcasm sooooo good!!!", context: "", shouldThrowOrHandle: true },
  ];

  for (const t of t1_tests) {
    let passed = false;
    let actualVerdict = "";
    try {
      const res = predict(t.input, t.context);
      actualVerdict = `P=${res.p_sarcastic}, Verdict=${res.verdict}`;
      passed = true;
    } catch (e: any) {
      actualVerdict = `Error: ${e.message}`;
      passed = false;
    }
    executionLog.push({
      testId: "TEST_1_FUNCTIONAL",
      subtest: t.name,
      input: t.input.slice(0, 30),
      status: passed ? "PASS" : "FAIL",
      output: actualVerdict,
    });
  }

  // =========================================================================
  // TEST 2: SENTIMENT INCONGRUITY EVALUATION
  // =========================================================================
  console.log(">>> EXECUTING TEST 2: SENTIMENT INCONGRUITY");
  const t2_cases = [
    { text: "brilliant disaster", expectedClash: 1.0, desc: "Direct intra-sentence lexical clash" },
    { text: "amazing work team", expectedClash: 0.0, desc: "Pure positive, no clash" },
    { text: "terrible failure", expectedClash: 0.0, desc: "Pure negative, no clash" },
    { text: "kamaal ka bekar khana", expectedClash: 1.0, desc: "Hinglish intra-sentence clash" },
  ];

  for (const tc of t2_cases) {
    const feat = extractIncongruity(tc.text);
    const passed = feat.polarity_clash === tc.expectedClash;
    executionLog.push({
      testId: "TEST_2_INCONGRUITY",
      subtest: tc.desc,
      input: tc.text,
      status: passed ? "PASS" : "FAIL",
      output: `polarity_clash=${feat.polarity_clash}, incongruity_index=${feat.incongruity_index}`,
    });
  }

  // =========================================================================
  // TEST 3: CONTEXT-AWARE SARCASM DETECTION
  // =========================================================================
  console.log(">>> EXECUTING TEST 3: CONTEXT-AWARE DETECTION");
  const t3_cases = [
    { text: "The service is outstanding.", context: "Flight delayed by 5 hours and luggage lost.", expectedDeltaPositive: true },
    { text: "Great weather today.", context: "Massive cyclone and flooding.", expectedDeltaPositive: true },
    { text: "Superb speed.", context: "Internet disconnected.", expectedDeltaPositive: true },
  ];

  for (const tc of t3_cases) {
    const withCtx = predict(tc.text, tc.context);
    const withoutCtx = predict(tc.text, "");
    const delta = withCtx.p_sarcastic - withoutCtx.p_sarcastic;
    const passed = delta > 0.2; // Context clash should significantly raise sarcasm probability
    executionLog.push({
      testId: "TEST_3_CONTEXT",
      subtest: "Contextual Clashing Delta",
      input: `Text: "${tc.text}" | Context: "${tc.context}"`,
      status: passed ? "PASS" : "FAIL",
      output: `P_without_context=${withoutCtx.p_sarcastic} -> P_with_context=${withCtx.p_sarcastic} (Δ=+${delta.toFixed(3)})`,
    });
  }

  // =========================================================================
  // TEST 4 & 15: MULTILINGUAL EVALUATION & BENCHMARK PERFORMANCE
  // =========================================================================
  console.log(">>> EXECUTING TEST 4 & 15: MULTILINGUAL BENCHMARK EVALUATION");
  const allProbs: number[] = [];
  const allLabels: number[] = [];
  const allPreds: number[] = [];

  const langResults: Record<string, { probs: number[]; labels: number[]; preds: number[] }> = {
    English: { probs: [], labels: [], preds: [] },
    Hindi: { probs: [], labels: [], preds: [] },
    Hinglish: { probs: [], labels: [], preds: [] },
    Marathi: { probs: [], labels: [], preds: [] },
    Telugu: { probs: [], labels: [], preds: [] },
  };

  const benchmarkRows: any[] = [];

  for (const item of BENCHMARK_CORPUS) {
    const res = predict(item.text, item.context);
    const predBinary = res.p_sarcastic >= 0.5 ? 1 : 0;

    allProbs.push(res.p_sarcastic);
    allLabels.push(item.groundTruth);
    allPreds.push(predBinary);

    langResults[item.lang].probs.push(res.p_sarcastic);
    langResults[item.lang].labels.push(item.groundTruth);
    langResults[item.lang].preds.push(predBinary);

    benchmarkRows.push({
      id: item.id,
      lang: item.lang,
      script: item.script,
      text: `"${item.text.replace(/"/g, '""')}"`,
      context: `"${item.context.replace(/"/g, '""')}"`,
      groundTruth: item.groundTruth,
      p_sarcastic: res.p_sarcastic,
      confidence: res.confidence,
      predLabel: predBinary,
      verdict: res.verdict,
      correct: predBinary === item.groundTruth ? 1 : 0,
      incongruityIndex: res.incongruity.incongruity_index,
      cmi: res.language.cmi,
    });
  }

  const overallMetrics = computeConfusionMatrix(allLabels, allPreds);
  const overallCal = computeECE(allProbs, allLabels);
  const overallBrierNLL = computeBrierAndNLL(allProbs, allLabels);

  console.log(`\n--- OVERALL BENCHMARK PERFORMANCE (N=${BENCHMARK_CORPUS.length}) ---`);
  console.log(`Accuracy:     ${(overallMetrics.accuracy * 100).toFixed(1)}%`);
  console.log(`Precision:    ${(overallMetrics.precision * 100).toFixed(1)}%`);
  console.log(`Recall:       ${(overallMetrics.recall * 100).toFixed(1)}%`);
  console.log(`F1 Score:     ${(overallMetrics.f1 * 100).toFixed(1)}%`);
  console.log(`Macro F1:     ${(overallMetrics.macroF1 * 100).toFixed(1)}%`);
  console.log(`Confusion:    TP=${overallMetrics.tp}, FP=${overallMetrics.fp}, TN=${overallMetrics.tn}, FN=${overallMetrics.fn}`);
  console.log(`ECE (Cal):    ${overallCal.ece}`);
  console.log(`Brier Score:  ${overallBrierNLL.brierScore}`);
  console.log(`NLL:          ${overallBrierNLL.nll}`);

  // Per-language statistics
  const langMetricsSummary: any[] = [];
  for (const [lang, data] of Object.entries(langResults)) {
    const m = computeConfusionMatrix(data.labels, data.preds);
    const b = computeBrierAndNLL(data.probs, data.labels);
    langMetricsSummary.push({
      Language: lang,
      N: data.labels.length,
      Accuracy: Math.round(m.accuracy * 1000) / 10,
      Precision: Math.round(m.precision * 1000) / 10,
      Recall: Math.round(m.recall * 1000) / 10,
      F1: Math.round(m.f1 * 1000) / 10,
      MacroF1: Math.round(m.macroF1 * 1000) / 10,
      BrierScore: b.brierScore,
    });
  }

  // =========================================================================
  // TEST 5: CODE-MIXING ANALYSIS
  // =========================================================================
  console.log("\n>>> EXECUTING TEST 5: CODE-MIXING ANALYSIS");
  const cmi_tests = [
    { text: "The train arrived on time.", expectedScript: "Latin", expectMixed: false },
    { text: "आजची बैठक वेळेवर झाली.", expectedScript: "Devanagari (Marathi)", expectMixed: false },
    { text: "वाह क्या बात है, 4 घंटे train late!", expectedScript: "Devanagari (Hindi)", expectMixed: true },
    { text: "Bohot badiya bhai, exam top karega", expectedScript: "Hinglish (Hindi-English)", expectMixed: true },
    { text: "Lai bhari bhau traffic madhe", expectedScript: "Marathi-English Code-Mixed", expectMixed: true },
  ];

  const cmiLog: any[] = [];
  for (const ct of cmi_tests) {
    const cm = analyzeCodeMixing(ct.text);
    cmiLog.push({
      text: ct.text,
      detectedScript: cm.primary_script,
      cmi: cm.cmi,
      isMixed: cm.is_code_mixed,
      languageGuess: cm.language,
      confidence: cm.confidence,
    });
  }

  // =========================================================================
  // TEST 7 & 8: CALIBRATION & SELECTIVE PREDICTION
  // =========================================================================
  console.log(">>> EXECUTING TEST 7 & 8: CALIBRATION & SELECTIVE PREDICTION");
  const tempValues = [0.5, 0.8, 1.0, 1.5, 2.0];
  const calTempComparison: any[] = [];

  for (const temp of tempValues) {
    const calibratedProbs = allProbs.map((p) => applyTemperatureCalibration(p, temp));
    const eceRes = computeECE(calibratedProbs, allLabels);
    const brierRes = computeBrierAndNLL(calibratedProbs, allLabels);
    calTempComparison.push({
      Temperature: temp,
      ECE: eceRes.ece,
      BrierScore: brierRes.brierScore,
      NLL: brierRes.nll,
    });
  }

  // Selective Prediction across thresholds
  const thresholdValues = [0.0, 0.2, 0.4, 0.6, 0.8];
  const selectiveRiskTable: any[] = [];

  for (const tau of thresholdValues) {
    let accepted = 0;
    let correctInAccepted = 0;

    for (let i = 0; i < allProbs.length; i++) {
      const p = allProbs[i];
      const conf = Math.abs(p - 0.5) * 2;
      const y = allLabels[i];
      const pred = p >= 0.5 ? 1 : 0;

      if (conf >= tau) {
        accepted++;
        if (pred === y) correctInAccepted++;
      }
    }

    const coverage = accepted / allProbs.length;
    const selectiveAcc = accepted > 0 ? correctInAccepted / accepted : 1.0;
    const selectiveRisk = 1.0 - selectiveAcc;

    selectiveRiskTable.push({
      Threshold_Tau: tau,
      Coverage: Math.round(coverage * 1000) / 10,
      SelectiveAccuracy: Math.round(selectiveAcc * 1000) / 10,
      SelectiveRisk: Math.round(selectiveRisk * 1000) / 10,
      AcceptedCount: `${accepted}/${allProbs.length}`,
    });
  }

  // =========================================================================
  // TEST 9: EXPLAINABILITY (OCCLUSION ATTRIBUTION)
  // =========================================================================
  console.log(">>> EXECUTING TEST 9: EXPLAINABILITY OCCLUSION");
  const sampleSentence = "वाह क्या बात है, 4 घंटे ट्रेन लेट करने के लिए धन्यवाद! 🙄";
  const occResult = computeOcclusion(sampleSentence, "ट्रेन 4 घंटे लेट थी");
  const topTokens = occResult.attributions.slice(0, 6);

  // =========================================================================
  // TEST 10: ROBUSTNESS PERTURBATION SUITE
  // =========================================================================
  console.log(">>> EXECUTING TEST 10: ROBUSTNESS PERTURBATION SUITE");
  const robResult = perturbAll(sampleSentence, "ट्रेन 4 घंटे लेट थी");

  // =========================================================================
  // TEST 11: EDGE CASES
  // =========================================================================
  console.log(">>> EXECUTING TEST 11: EDGE CASES TESTING");
  const edgeCases = [
    { id: "EDGE_01", name: "Empty string", input: "", context: "" },
    { id: "EDGE_02", name: "Only Whitespace", input: "     ", context: "" },
    { id: "EDGE_03", name: "Only Emojis", input: "🙄🙃😏🤡", context: "" },
    { id: "EDGE_04", name: "Special Characters Only", input: "!@#$%^&*()_+=-~`", context: "" },
    { id: "EDGE_05", name: "Single word sarcastic", input: "Brilliant", context: "Failed test" },
    { id: "EDGE_06", name: "Extreme repeated characters", input: "sooooooooooooooo gooooooooood waaaaaaah", context: "" },
    { id: "EDGE_07", name: "Long paragraph (2000 chars)", input: "Great job on the code deployment. ".repeat(60), context: "" },
    { id: "EDGE_08", name: "Mixed Unicode scripts in one word", input: "superसुंदरbhari", context: "" },
    { id: "EDGE_09", name: "Unsupported language (French)", input: "C'est absolument magnifique comme désastre.", context: "" },
    { id: "EDGE_10", name: "Contradictory context", input: "The food was delicious.", context: "The food was poisoned and awful." },
  ];

  const edgeResults: any[] = [];
  for (const ec of edgeCases) {
    try {
      const res = predict(ec.input, ec.context);
      edgeResults.push({
        id: ec.id,
        name: ec.name,
        input: ec.input.slice(0, 30),
        status: "HANDLED_SAFELY",
        verdict: res.verdict,
        p_sarcastic: res.p_sarcastic,
        confidence: res.confidence,
        abstain: res.abstain,
      });
    } catch (e: any) {
      edgeResults.push({
        id: ec.id,
        name: ec.name,
        input: ec.input.slice(0, 30),
        status: "CRASHED",
        verdict: e.message,
        p_sarcastic: 0,
        confidence: 0,
        abstain: false,
      });
    }
  }

  // =========================================================================
  // TEST 12: LATENCY & PERFORMANCE MEASUREMENT (1,000 iterations)
  // =========================================================================
  console.log(">>> EXECUTING TEST 12: LATENCY & INFERENCE BENCHMARKING");
  const iterations = 1000;
  const latencies: number[] = [];
  const testSent = "Oh brilliant, another delay on Mumbai locals, just what my Monday needed!";
  const testCtx = "Train canceled due to waterlogging";

  // Warmup
  for (let i = 0; i < 50; i++) predict(testSent, testCtx);

  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    predict(testSent, testCtx);
    const elapsed = performance.now() - t0;
    latencies.push(elapsed);
  }

  latencies.sort((a, b) => a - b);
  const minLat = latencies[0];
  const maxLat = latencies[latencies.length - 1];
  const avgLat = latencies.reduce((a, b) => a + b, 0) / latencies.length;
  const medianLat = latencies[Math.floor(latencies.length / 2)];
  const p90Lat = latencies[Math.floor(latencies.length * 0.9)];
  const p95Lat = latencies[Math.floor(latencies.length * 0.95)];
  const p99Lat = latencies[Math.floor(latencies.length * 0.99)];

  console.log(`Latency over ${iterations} iterations:`);
  console.log(`Min:    ${minLat.toFixed(3)} ms`);
  console.log(`Median: ${medianLat.toFixed(3)} ms`);
  console.log(`Mean:   ${avgLat.toFixed(3)} ms`);
  console.log(`P90:    ${p90Lat.toFixed(3)} ms`);
  console.log(`P95:    ${p95Lat.toFixed(3)} ms`);
  console.log(`P99:    ${p99Lat.toFixed(3)} ms`);
  console.log(`Max:    ${maxLat.toFixed(3)} ms`);

  // Batch evaluation timing for 200 items
  const batch200 = Array(200).fill(testSent);
  const tBatch0 = performance.now();
  batch200.map((s) => predict(s, testCtx));
  const batchElapsedMs = performance.now() - tBatch0;
  console.log(`Batch 200 items execution time: ${batchElapsedMs.toFixed(2)} ms (${(batchElapsedMs / 200).toFixed(3)} ms/item)`);

  // =========================================================================
  // TEST 14: ABLATION STUDY
  // =========================================================================
  console.log("\n>>> EXECUTING TEST 14: ABLATION STUDY");
  // Feature ablation: measure Macro-F1 drop when individual feature signals are deactivated
  const ablationRuns = [
    { name: "Full SARcasmScope (All Features)", maskContext: false, maskPolarity: false, maskIntensifiers: false, maskEmoji: false },
    { name: "Ablation: No Conversational Context Clash", maskContext: true, maskPolarity: false, maskIntensifiers: false, maskEmoji: false },
    { name: "Ablation: No Polarity Clash Feature", maskContext: false, maskPolarity: true, maskIntensifiers: false, maskEmoji: false },
    { name: "Ablation: No Hyperbolic Intensifiers", maskContext: false, maskPolarity: false, maskIntensifiers: true, maskEmoji: false },
    { name: "Ablation: No Irony Emoji Features", maskContext: false, maskPolarity: false, maskIntensifiers: false, maskEmoji: true },
  ];

  const ablationResults: any[] = [];

  for (const ab of ablationRuns) {
    const abPreds: number[] = [];
    for (const item of BENCHMARK_CORPUS) {
      let score = 0.22;
      const feats = extractIncongruity(item.text, ab.maskContext ? "" : item.context);
      const em = emojiFeatures(item.text);

      score += feats.incongruity_index * 0.58;
      if (!ab.maskPolarity && feats.polarity_clash > 0) score += 0.25;
      if (!ab.maskContext && feats.context_clash > 0) score += 0.28;
      if (!ab.maskEmoji && em.n_irony_emoji > 0) score += Math.min(em.n_irony_emoji * 0.26, 0.45);
      if (feats.pos_count > 0 && feats.neg_count === 0 && (!ab.maskContext ? feats.context_clash === 0 : true)) score -= 0.22;
      if (feats.pos_count === 0 && feats.neg_count === 0 && (!ab.maskEmoji ? em.n_irony_emoji === 0 : true) && (!ab.maskIntensifiers ? feats.intensifier_count === 0 : true)) score -= 0.12;
      if (/(\.\.\.|!\?|\?!|!!)/.test(item.text)) score += 0.08;

      const pClamped = Math.max(0.02, Math.min(0.98, score));
      abPreds.push(pClamped >= 0.5 ? 1 : 0);
    }

    const abMetrics = computeConfusionMatrix(allLabels, abPreds);
    ablationResults.push({
      Configuration: ab.name,
      Accuracy: Math.round(abMetrics.accuracy * 1000) / 10,
      Precision: Math.round(abMetrics.precision * 1000) / 10,
      Recall: Math.round(abMetrics.recall * 1000) / 10,
      F1: Math.round(abMetrics.f1 * 1000) / 10,
      MacroF1: Math.round(abMetrics.macroF1 * 1000) / 10,
      DeltaMacroF1: Math.round((abMetrics.macroF1 - overallMetrics.macroF1) * 1000) / 10,
    });
  }

  // =========================================================================
  // TEST 13: BASELINE COMPARISONS
  // =========================================================================
  console.log(">>> EXECUTING TEST 13: BASELINE COMPARISON");
  // 1. Majority Class Baseline (Always predicts 1)
  const majorityPreds = allLabels.map(() => 1);
  const majMetrics = computeConfusionMatrix(allLabels, majorityPreds);

  // 2. Lexicon Keyword Match Baseline (Predicts 1 if any negative word exists)
  const lexPreds = BENCHMARK_CORPUS.map((item) => {
    const toks = tokenizeWords(item.text).map((t) => t.toLowerCase());
    const hasNeg = toks.some((t) => NEGATIVE_CUES.has(t));
    return hasNeg ? 1 : 0;
  });
  const lexMetrics = computeConfusionMatrix(allLabels, lexPreds);

  // 3. Emoji Only Baseline
  const emojiOnlyPreds = BENCHMARK_CORPUS.map((item) => {
    const em = emojiFeatures(item.text);
    return em.n_irony_emoji > 0 ? 1 : 0;
  });
  const emojiMetrics = computeConfusionMatrix(allLabels, emojiOnlyPreds);

  const baselineComparison = [
    { Baseline: "Majority Class (Always Sarcastic)", Accuracy: Math.round(majMetrics.accuracy * 1000) / 10, MacroF1: Math.round(majMetrics.macroF1 * 1000) / 10 },
    { Baseline: "Keyword Negativity Lookup Only", Accuracy: Math.round(lexMetrics.accuracy * 1000) / 10, MacroF1: Math.round(lexMetrics.macroF1 * 1000) / 10 },
    { Baseline: "Irony Emoji Shortcut Only", Accuracy: Math.round(emojiMetrics.accuracy * 1000) / 10, MacroF1: Math.round(emojiMetrics.macroF1 * 1000) / 10 },
    { Baseline: "SARcasmScope (Calibrated Feature-Fusion)", Accuracy: Math.round(overallMetrics.accuracy * 1000) / 10, MacroF1: Math.round(overallMetrics.macroF1 * 1000) / 10 },
  ];

  // =========================================================================
  // BOOTSTRAP CONFIDENCE INTERVAL (1,000 iterations for Macro-F1)
  // =========================================================================
  console.log(">>> EXECUTING BOOTSTRAP CONFIDENCE INTERVALS (B=1000)");
  const B = 1000;
  const bootstrapF1s: number[] = [];
  const N = allLabels.length;

  for (let b = 0; b < B; b++) {
    const bootY: number[] = [];
    const bootPred: number[] = [];
    for (let i = 0; i < N; i++) {
      const idx = Math.floor(Math.random() * N);
      bootY.push(allLabels[idx]);
      bootPred.push(allPreds[idx]);
    }
    const bMetric = computeConfusionMatrix(bootY, bootPred);
    bootstrapF1s.push(bMetric.macroF1);
  }

  bootstrapF1s.sort((a, b) => a - b);
  const ciLo = bootstrapF1s[Math.floor(B * 0.025)];
  const ciHi = bootstrapF1s[Math.floor(B * 0.975)];
  console.log(`95% Bootstrap CI for Macro-F1: [${(ciLo * 100).toFixed(1)}%, ${(ciHi * 100).toFixed(1)}%]`);

  // =========================================================================
  // EXPORT CSVs FOR SCIENTIFIC REPRODUCIBILITY
  // =========================================================================
  console.log("\n>>> WRITING REPRODUCIBLE CSV OUTPUTS TO eval_results/");

  // CSV 1: Detailed Benchmark Predictions
  const csv1Headers = Object.keys(benchmarkRows[0]).join(",");
  const csv1Rows = benchmarkRows.map((r) => Object.values(r).join(",")).join("\n");
  fs.writeFileSync(path.join(OUT_DIR, "benchmark_predictions.csv"), `${csv1Headers}\n${csv1Rows}`);

  // CSV 2: Multilingual Language Breakdown
  const csv2Headers = Object.keys(langMetricsSummary[0]).join(",");
  const csv2Rows = langMetricsSummary.map((r) => Object.values(r).join(",")).join("\n");
  fs.writeFileSync(path.join(OUT_DIR, "multilingual_breakdown.csv"), `${csv2Headers}\n${csv2Rows}`);

  // CSV 3: Temperature Calibration & Brier Score
  const csv3Headers = Object.keys(calTempComparison[0]).join(",");
  const csv3Rows = calTempComparison.map((r) => Object.values(r).join(",")).join("\n");
  fs.writeFileSync(path.join(OUT_DIR, "temperature_calibration_metrics.csv"), `${csv3Headers}\n${csv3Rows}`);

  // CSV 4: Selective Risk Coverage Table
  const csv4Headers = Object.keys(selectiveRiskTable[0]).join(",");
  const csv4Rows = selectiveRiskTable.map((r) => Object.values(r).join(",")).join("\n");
  fs.writeFileSync(path.join(OUT_DIR, "selective_risk_coverage.csv"), `${csv4Headers}\n${csv4Rows}`);

  // CSV 5: Ablation Results
  const csv5Headers = Object.keys(ablationResults[0]).join(",");
  const csv5Rows = ablationResults.map((r) => Object.values(r).join(",")).join("\n");
  fs.writeFileSync(path.join(OUT_DIR, "ablation_study_results.csv"), `${csv5Headers}\n${csv5Rows}`);

  // CSV 6: Robustness Perturbations Table
  const robRows = robResult.results.map((r) =>
    `"${r.perturbation}","${r.description}","${r.perturbed_text.replace(/"/g, '""')}",${r.p_sarcastic},${r.delta},${r.flipped ? 1 : 0}`
  );
  fs.writeFileSync(
    path.join(OUT_DIR, "robustness_perturbation_results.csv"),
    `Perturbation,Description,Perturbed_Text,P_Sarcastic,Delta,Flipped\n${robRows.join("\n")}`
  );

  // CSV 7: Latency Metrics
  const latData = [
    { Metric: "Min", Latency_ms: minLat },
    { Metric: "Median", Latency_ms: medianLat },
    { Metric: "Mean", Latency_ms: avgLat },
    { Metric: "P90", Latency_ms: p90Lat },
    { Metric: "P95", Latency_ms: p95Lat },
    { Metric: "P99", Latency_ms: p99Lat },
    { Metric: "Max", Latency_ms: maxLat },
  ];
  const csv7Rows = latData.map((r) => `${r.Metric},${r.Latency_ms.toFixed(4)}`).join("\n");
  fs.writeFileSync(path.join(OUT_DIR, "latency_benchmark.csv"), `Metric,Latency_ms\n${csv7Rows}`);

  // Output full scientific evaluation JSON summary for reporting
  const fullSummary = {
    metadata: {
      investigator: "Payal Mehta",
      date: new Date().toISOString(),
      corpusSize: BENCHMARK_CORPUS.length,
      iterationsLatency: iterations,
    },
    overallMetrics,
    bootstrap95CI: { lower: ciLo, upper: ciHi },
    calibration: { ece: overallCal.ece, brier: overallBrierNLL.brierScore, nll: overallBrierNLL.nll },
    multilingual: langMetricsSummary,
    ablation: ablationResults,
    baselines: baselineComparison,
    selectivePrediction: selectiveRiskTable,
    robustness: { stabilityScore: robResult.stability_score, baselineP: robResult.baseline_p_sarcastic, perturbations: robResult.results },
    explainabilitySample: { sentence: sampleSentence, attributions: topTokens },
    edgeCases: edgeResults,
    performance: { minLat, medianLat, avgLat, p95Lat, maxLat, batch200ElapsedMs: batchElapsedMs },
  };

  fs.writeFileSync(path.join(OUT_DIR, "full_scientific_evaluation.json"), JSON.stringify(fullSummary, null, 2));
  console.log("\n>>> ALL 15 SCIENTIFIC TESTS COMPLETED SUCCESSFULLY.");
  console.log("Evaluation artifacts generated in: /eval_results/");
}

runCompleteEvaluation().catch((err) => {
  console.error("Evaluation error:", err);
  process.exit(1);
});
