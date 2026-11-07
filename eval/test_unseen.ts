import { predict } from "../src/engine/sarcasmEngine";

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

console.log("=== UNSEEN GENERALIZATION EVALUATION ===");
let correct = 0;
let tp = 0, fp = 0, tn = 0, fn = 0;

for (const tc of UNSEEN_TEST_CASES) {
  const p = predict(tc.text, tc.context);
  const isSarc = p.p_sarcastic >= 0.5 ? 1 : 0;
  const match = isSarc === tc.expected;
  if (match) correct++;
  if (tc.expected === 1 && isSarc === 1) tp++;
  else if (tc.expected === 0 && isSarc === 1) fp++;
  else if (tc.expected === 0 && isSarc === 0) tn++;
  else if (tc.expected === 1 && isSarc === 0) fn++;

  console.log(`[${match ? "PASS" : "FAIL"}] Expected: ${tc.expected} | Got: ${isSarc} (P=${p.p_sarcastic.toFixed(3)}) | ID: ${tc.id} (${tc.lang})`);
}

const acc = (correct / UNSEEN_TEST_CASES.length) * 100;
const prec = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
const rec = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;

console.log(`\nUnseen Test Accuracy:  ${correct}/${UNSEEN_TEST_CASES.length} (${acc.toFixed(1)}%)`);
console.log(`Unseen Sarcasm Precision: ${prec.toFixed(1)}%`);
console.log(`Unseen Sarcasm Recall:    ${rec.toFixed(1)}%`);
console.log(`Unseen F1 Score:          ${f1.toFixed(1)}%`);
console.log(`Unseen Confusion Matrix:  TP=${tp}, FP=${fp}, TN=${tn}, FN=${fn}`);
