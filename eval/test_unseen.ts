import { predict as predictLinguistic } from "../src/engine/sarcasmEngine";
import {
  predictHybrid,
  DEFAULT_ABSTAIN_THRESHOLD,
} from "../src/engine/hybridFusion";

export const UNSEEN_TEST_CASES = [
  // --- Unseen Sarcastic (5 items across languages) ---
  {
    id: "UNSEEN_SARC_01",
    lang: "English",
    text: "Oh fantastic, the electricity went out right during my final online exam.",
    context: "Power outage during test",
    expected: 1,
    desc: "Praise 'fantastic' under exam disaster",
  },
  {
    id: "UNSEEN_SARC_02",
    lang: "Hinglish",
    text: "Bohot khoob, 5 ghante late flight ke baad bag bhi nahi mila! 🙄",
    context: "Lost luggage at airport",
    expected: 1,
    desc: "Hinglish sarcasm 'Bohot khoob' with irony emoji and luggage loss",
  },
  {
    id: "UNSEEN_SARC_03",
    lang: "Marathi",
    text: "खूप छान, ऐन पावसाळ्यात छत्री घरीच विसरून आलो!",
    context: "पावसात भिजलो",
    expected: 1,
    desc: "Marathi Devanagari praise 'खूप छान' for forgetting umbrella in rain",
  },
  {
    id: "UNSEEN_SARC_04",
    lang: "Marathi-Latin",
    text: "Wah re wah, navin phone cover pahilyach divshi tutla!",
    context: "Damage to phone",
    expected: 1,
    desc: "Rhetorical 'Wah re wah' for broken phone cover",
  },
  {
    id: "UNSEEN_SARC_05",
    lang: "Telugu",
    text: "వావ్ అద్భుతం, ఎగ్జామ్ హాల్‌కి వెళ్లేటప్పుడు బైక్ పంచర్ అయింది 🙃",
    context: "టైర్ పంచర్",
    expected: 1,
    desc: "Telugu sarcasm 'వావ్ అద్భుతం' for flat tire before exam with 🙃",
  },

  // --- Unseen Literal: Direct Complaints (5 items - MUST NOT be falsely flagged!) ---
  {
    id: "UNSEEN_LIT_01",
    lang: "English",
    text: "The battery life on this phone is terrible and drains in two hours.",
    context: "Battery issue",
    expected: 0,
    desc: "Direct literal negative complaint, no irony",
  },
  {
    id: "UNSEEN_LIT_02",
    lang: "Marathi",
    text: "ट्रेन रद्द झाली त्यामुळे आम्हाला हॉटेलमध्ये राहावे लागले.",
    context: "प्रवासात अडचण",
    expected: 0,
    desc: "Factual literal delay/cancellation report in Marathi",
  },
  {
    id: "UNSEEN_LIT_03",
    lang: "English",
    text: "Internet connection is completely down since morning.",
    context: "ISP breakdown",
    expected: 0,
    desc: "Literal network failure statement",
  },
  {
    id: "UNSEEN_LIT_04",
    lang: "Hinglish",
    text: "Khana thanda tha aur bilkul swaad nahi tha.",
    context: "Bad restaurant dinner",
    expected: 0,
    desc: "Direct complaint about food in Hinglish",
  },
  {
    id: "UNSEEN_LIT_05",
    lang: "Telugu",
    text: "రైలు చాలా ఆలస్యంగా నడుస్తోంది, చాలా ఇబ్బందిగా ఉంది.",
    context: "ఆలస్యం",
    expected: 0,
    desc: "Literal complaint about late train in Telugu",
  },

  // --- Unseen Literal: Sincere Praise (5 items - MUST NOT be falsely flagged!) ---
  {
    id: "UNSEEN_LIT_06",
    lang: "English",
    text: "The hotel room was spotless and the staff was very polite and helpful.",
    context: "Family vacation",
    expected: 0,
    desc: "Sincere hospitality praise",
  },
  {
    id: "UNSEEN_LIT_07",
    lang: "Marathi",
    text: "आजचा निकाल खूप चांगला लागला, आई बाबा खूप खूश झाले!",
    context: "परीक्षेत यश",
    expected: 0,
    desc: "Sincere academic success praise in Marathi",
  },
  {
    id: "UNSEEN_LIT_08",
    lang: "English",
    text: "We received our shipment 2 days ahead of schedule, thank you!",
    context: "Fast delivery service",
    expected: 0,
    desc: "Sincere delivery appreciation",
  },
  {
    id: "UNSEEN_LIT_09",
    lang: "Hindi",
    text: "नया लैपटॉप बहुत स्मूथ और फास्ट चल रहा है, बहुत खुशी हुई।",
    context: "नया गैजेट",
    expected: 0,
    desc: "Sincere tech praise in Hindi",
  },
  {
    id: "UNSEEN_LIT_10",
    lang: "Telugu",
    text: "సినిమా పాటలు చాలా అద్భుతంగా ఉన్నాయి, నాకు బాగా నచ్చాయి.",
    context: "మంచి సంగీతం",
    expected: 0,
    desc: "Sincere music review in Telugu",
  },
];

export function evaluateUnseenSuite() {
  let lingTP = 0, lingFP = 0, lingTN = 0, lingFN = 0;
  let hybTP = 0, hybFP = 0, hybTN = 0, hybFN = 0;
  let hybAbstain = 0;

  console.log("=== UNSEEN GENERALIZATION EVALUATION (N=15) ===");
  for (const tc of UNSEEN_TEST_CASES) {
    const ling = predictLinguistic(tc.text, tc.context);
    const hyb = predictHybrid(tc.text, tc.context);

    const lingPred = ling.p_sarcastic >= 0.5 ? 1 : 0;
    const hybPred = hyb.p_fused >= 0.5 ? 1 : 0;
    if (hyb.abstain) hybAbstain++;

    if (tc.expected === 1 && lingPred === 1) lingTP++;
    else if (tc.expected === 0 && lingPred === 1) lingFP++;
    else if (tc.expected === 0 && lingPred === 0) lingTN++;
    else lingFN++;

    if (tc.expected === 1 && hybPred === 1) hybTP++;
    else if (tc.expected === 0 && hybPred === 1) hybFP++;
    else if (tc.expected === 0 && hybPred === 0) hybTN++;
    else hybFN++;

    console.log(
      `[Expected:${tc.expected}] Ling=${lingPred}(${ling.p_sarcastic.toFixed(3)}) | Hybrid=${hybPred}(${hyb.p_fused.toFixed(3)}, conf=${hyb.confidence.toFixed(2)}, ${hyb.verdict}) | ${tc.id} (${tc.lang})`
    );
  }

  const calc = (tp: number, fp: number, tn: number, fn: number) => {
    const n = tp + fp + tn + fn;
    const acc = ((tp + tn) / n) * 100;
    const p1 = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
    const r1 = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
    const f1 = p1 + r1 > 0 ? (2 * p1 * r1) / (p1 + r1) : 0;
    const p0 = tn + fn > 0 ? (tn / (tn + fn)) * 100 : 0;
    const r0 = tn + fp > 0 ? (tn / (tn + fp)) * 100 : 0;
    const f0 = p0 + r0 > 0 ? (2 * p0 * r0) / (p0 + r0) : 0;
    const mf1 = (f1 + f0) / 2;
    return { acc, p1, r1, f1, mf1, tp, fp, tn, fn };
  };

  const mLing = calc(lingTP, lingFP, lingTN, lingFN);
  const mHyb = calc(hybTP, hybFP, hybTN, hybFN);

  console.log(`\nStage 2 Linguistic: Acc=${mLing.acc.toFixed(1)}% Prec=${mLing.p1.toFixed(1)}% Rec=${mLing.r1.toFixed(1)}% MF1=${mLing.mf1.toFixed(1)}% (TP=${mLing.tp}, FP=${mLing.fp}, TN=${mLing.tn}, FN=${mLing.fn})`);
  console.log(`Calibrated Hybrid:  Acc=${mHyb.acc.toFixed(1)}% Prec=${mHyb.p1.toFixed(1)}% Rec=${mHyb.r1.toFixed(1)}% MF1=${mHyb.mf1.toFixed(1)}% (TP=${mHyb.tp}, FP=${mHyb.fp}, TN=${mHyb.tn}, FN=${mHyb.fn}, Abstained @ tau=${DEFAULT_ABSTAIN_THRESHOLD}: ${hybAbstain}/${UNSEEN_TEST_CASES.length})`);

  return { mLing, mHyb, hybAbstain };
}

const isMain =
  typeof process !== "undefined" &&
  process.argv[1] &&
  (process.argv[1].endsWith("test_unseen.ts") || process.argv[1].endsWith("test_unseen.js"));

if (isMain) {
  evaluateUnseenSuite();
}
