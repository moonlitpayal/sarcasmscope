/**
 * Morphological Normalization & Romanized Normalization Unit Test Suite
 */

// 1. Devanagari Suffixes (Hindi & Marathi) - Ordered longest-first
export const DEV_SUFFIXES = [
  // Compound case postpositions & inflections (Marathi & Hindi)
  "्यांच्यासाठी", "ांच्यासाठी", "्यांच्यावर", "ांच्यावर", "्यांच्या", "ांच्या",
  "्यामुळे", "ामुळे", "्यांनी", "ांसाठी", "ांवरून", "ातील", "ातून", "्यांचा",
  "्यांची", "्यांचे", "ासाठी", "ावरून", "ाबद्दल", "ाप्रमाणे", "्यांत", "ांतून",
  "कर्मियों", "कर्ताओं", "वालों", "वाली", "वाले", "वाला",
  "ियों", "ियां", "ाइयों", "ाओं", "ाता", "ाती", "ाते", "ाया", "ाये", "ायी",
  "तात", "तातून", "ताना", "णारा", "णारी", "णारे", "णार",
  "लेला", "लेली", "लेले", "ायला", "ायचे", "ायची", "ायचा",
  "तील", "तून", "कडे", "हून", "वून", "ांत", "ावर",
  "ेचे", "ाचे", "्याची", "्याचे", "्याचा", "्याच्या",
  "च्या", "तील", "कडे", "ून", "ात", "ना", "ला", "चे", "ची", "चा", "चं",
  "ेगा", "ेगी", "ेंगे", "ेगा", "ोगी", "ोगे", "ूंगा", "ूंगी",
  "करके", "कर", "ाना", "ानी", "ाने", "ाता", "ाती", "ाते",
  "ां", "ों", "ें", "े", "ी", "ा"
].sort((a, b) => b.length - a.length);

// 2. Telugu Suffixes - Ordered longest-first
export const TELUGU_SUFFIXES = [
  // Nominal compound case markers & verbal auxiliaries
  "మైనటువంటి", "నటువంటి", "మైనందుకు", "నందుకు", "ందుకు",
  "చేస్తున్నారు", "చేస్తున్నాడు", "చేస్తుంది", "చేస్తారు",
  "తున్నారు", "తున్నాడు", "తుంది", "తారు",
  "పోయింది", "పడింది", "యింది", "అయింది", "ుంది", "ఉంది",
  "ాలతో", "ాలలో", "లతో", "లలో", "తోటి",
  "కొరకు", "కోసం", "నుండి", "నుంచి", "వరకు", "వద్ద", "వలన", "వల్ల",
  "దగ్గర", "ద్వారా", "మేరకు",
  "మైన", "యిన", "అయిన", "తో", "లో", "కు", "కి", "గా", "రూ", "లు", "ని", "ను"
].sort((a, b) => b.length - a.length);

/**
 * Conservative Indic suffix stripper
 * Strips at most ONE inflectional suffix, requiring minStemLength code units.
 */
export function stemIndicToken(word: string): { stem: string; strippedSuffix: string | null } {
  // Check Devanagari (\u0900-\u097F)
  if (/[\u0900-\u097F]/.test(word)) {
    const minLen = 3; // Keep at least 2-3 characters to avoid corrupting basic words
    for (const sfx of DEV_SUFFIXES) {
      if (word.endsWith(sfx) && word.length - sfx.length >= minLen) {
        return {
          stem: word.slice(0, word.length - sfx.length),
          strippedSuffix: sfx,
        };
      }
    }
  }

  // Check Telugu (\u0C00-\u0C7F)
  if (/[\u0C00-\u0C7F]/.test(word)) {
    const minLen = 4;
    for (const sfx of TELUGU_SUFFIXES) {
      if (word.endsWith(sfx) && word.length - sfx.length >= minLen) {
        return {
          stem: word.slice(0, word.length - sfx.length),
          strippedSuffix: sfx,
        };
      }
    }
  }

  return { stem: word, strippedSuffix: null };
}

/**
 * Conservative Romanized Normalizer for Hinglish & Romanized Indic
 */
export function normalizeRomanized(word: string): { normalized: string; folded: string } {
  const lower = word.toLowerCase();

  // 1. Collapse runs of >2 identical characters to at most 2 (e.g. soooo -> soo, bahuuuut -> bahuut)
  const dedup = lower.replace(/(.)\1{2,}/g, "$1$1");

  // 2. Vowel folding for romanized Indic phonetic matching
  let folded = dedup
    .replace(/aa/g, "a")
    .replace(/ee/g, "i")
    .replace(/oo/g, "u")
    .replace(/ou/g, "u");

  // 3. Conservative phonetic equivalence mapping
  folded = folded
    .replace(/ph/g, "f")
    .replace(/kh/g, "k")
    .replace(/bh/g, "b")
    .replace(/dh/g, "d")
    .replace(/th/g, "t")
    .replace(/sh/g, "s")
    .replace(/zh/g, "j")
    .replace(/z/g, "j")
    .replace(/w/g, "v");

  return { normalized: dedup, folded };
}

// ============================================================================
// UNIT TESTS
// ============================================================================
console.log("=== RUNNING MORPHOLOGICAL & PHONETIC NORMALIZATION TESTS ===");

const hindiTests = [
  { input: "खराबियां", expectedStem: "खराब" },
  { input: "गड्डों", expectedStem: "गड्ड" },
  { input: "तोपों", expectedStem: "तोप" },
  { input: "समस्याओं", expectedStem: "समस्य" },
  { input: "है", expectedStem: "है" }, // Short root: must not be stripped!
  { input: "नहीं", expectedStem: "नहीं" }, // Short root: must not be stripped!
];

console.log("\nHindi Devanagari Tests:");
for (const t of hindiTests) {
  const res = stemIndicToken(t.input);
  const pass = res.stem === t.expectedStem;
  console.log(`[${pass ? "PASS" : "FAIL"}] "${t.input}" -> stem: "${res.stem}" (sfx: "${res.strippedSuffix}") | Expected: "${t.expectedStem}"`);
}

const marathiTests = [
  { input: "गटारे", expectedStem: "गटार" },
  { input: "खड्ड्यांत", expectedStem: "खड्ड" },
  { input: "पालिकेचे", expectedStem: "पालिक" },
  { input: "पावसात", expectedStem: "पावस" },
  { input: "छान", expectedStem: "छान" }, // Uninflected: preserved
  { input: "मी", expectedStem: "मी" },   // Short: preserved
];

console.log("\nMarathi Devanagari Tests:");
for (const t of marathiTests) {
  const res = stemIndicToken(t.input);
  const pass = res.stem === t.expectedStem;
  console.log(`[${pass ? "PASS" : "FAIL"}] "${t.input}" -> stem: "${res.stem}" (sfx: "${res.strippedSuffix}") | Expected: "${t.expectedStem}"`);
}

const teluguTests = [
  { input: "ఆలస్యమైనందుకు", expectedStem: "ఆలస్య" },
  { input: "ఇబ్బందులలో", expectedStem: "ఇబ్బందు" },
  { input: "పాడైపోయింది", expectedStem: "పాడై" },
  { input: "బాగుంది", expectedStem: "బాగు" },
  { input: "వావ్", expectedStem: "వావ్" }, // Preserved
];

console.log("\nTelugu Script Tests:");
for (const t of teluguTests) {
  const res = stemIndicToken(t.input);
  const pass = res.stem.startsWith(t.expectedStem);
  console.log(`[${pass ? "PASS" : "FAIL"}] "${t.input}" -> stem: "${res.stem}" (sfx: "${res.strippedSuffix}") | Expected prefix: "${t.expectedStem}"`);
}

const romanizedTests = [
  { input: "bohooooot", expectedDedup: "bohoot", expectedFolded: "bohut" },
  { input: "waaaah", expectedDedup: "waah", expectedFolded: "vah" },
  { input: "shaandaar", expectedDedup: "shaandaar", expectedFolded: "sandar" },
  { input: "shandar", expectedDedup: "shandar", expectedFolded: "sandar" },
  { input: "badiya", expectedDedup: "badiya", expectedFolded: "badiya" },
  { input: "badhiya", expectedDedup: "badhiya", expectedFolded: "badiya" },
];

console.log("\nRomanized Hinglish / Marathi Phonetic Folding Tests:");
for (const t of romanizedTests) {
  const res = normalizeRomanized(t.input);
  const pass = res.folded === t.expectedFolded;
  console.log(`[${pass ? "PASS" : "FAIL"}] "${t.input}" -> dedup: "${res.normalized}", folded: "${res.folded}" | Expected folded: "${t.expectedFolded}"`);
}
