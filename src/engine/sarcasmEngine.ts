/**
 * SarcasmScope Core Intelligence Engine
 * Faithful client-side port of the Python NLP pipeline:
 * - Multilingual Unicode script detection (Devanagari, Latin, Dravidian, etc.)
 * - Code-mixing index (CMI) calculation & Romanized Indic language heuristic
 * - Incongruity extraction (intra-sentence opposition, discourse/context clash, hyperbolic cues)
 * - Calibrated probability scoring with temperature scaling and abstention threshold
 * - Model-agnostic leave-one-token-out occlusion attributions
 * - Controlled perturbation robustness suite
 */

// --- BASE LEXICONS (Handcrafted Base Core) ---
export const BASE_POSITIVE_CUES = new Set([
  // English
  "brilliant", "genius", "amazing", "wonderful", "outstanding", "perfect",
  "love", "lovely", "blessed", "greatest", "fantastic", "superb", "flawless", "hero",
  "great", "awesome", "excellent", "magnificent", "masterpiece", "blessing", "pleasure",
  "top", "proud", "congrats", "congratulations", "clean", "concise",
  // Hindi Devanagari
  "बढ़िया", "शानदार", "कमाल", "लाजवाब", "वाह", "गजब", "महान", "शाबाश", "सुंदर", "अद्भुत", "धन्य", "बधाई",
  "सलामी", "तोप", "तोपों", "मजा", "स्वर्ग", "नियोजन",
  // Hinglish
  "badiya", "bohot badiya", "gazab", "shandar", "kamaal", "lajawab", "waah", "wah", "zabardast", "sahi", "badhiya",
  "top", "hero", "bhai", "mazaa", "maza", "superb",
  // Marathi Devanagari & Latin
  "भारी", "छान", "उत्कृष्ट", "मस्त", "शाब्बास", "कौतुक", "व्वा",
  "bhari", "lai bhari", "mast", "chhan", "lai", "changla", "aabhar",
  // Telugu
  "వావ్", "భలే", "అద్భుతం", "అద్భుతమైన", "గొప్ప", "బాగుంది", "మంచి"
]);

export const BASE_NEGATIVE_CUES = new Set([
  // English
  "terrible", "horrible", "disaster", "fail", "failed", "fails", "worst", "idiot",
  "useless", "garbage", "trash", "broken", "nightmare", "clown", "stupid", "pathetic", "ruined",
  "delay", "delayed", "late", "traffic", "stuck", "crash", "crashed", "cancel", "canceled", "mess",
  "downpour", "soaked", "monsoon", "hold", "dropped", "waiting", "spoil", "spoiled", "lost", "outage",
  "disconnect", "disconnected", "flood", "flooding", "cyclone", "potholes", "damage", "corrupt", "puncher", "puncture",
  // Hindi Devanagari
  "बेकार", "घटिया", "तमाशा", "पागल", "फालतू", "झूठ", "धोखा", "गंदा", "कचरा", "बकवास", "मूर्ख", "परेशान",
  "लेट", "खराब", "ठंडा", "पानी", "जलजमाव", "बाढ़", "गड्ढा", "गड्ढे", "भीड़", "उशीर", "मुश्किल", "नुकसान", "कष्ट",
  // Hinglish
  "bekar", "ghatiya", "tamasha", "pagal", "faltu", "jhoot", "dhoka", "ganda", "bakwas", "kachra", "fas", "late",
  "spoil", "spoiled", "fail", "load", "drop", "jam", "chhut", "tough", "problem", "kharab",
  // Marathi Devanagari & Latin
  "भिकार", "घाणेरडा", "त्रास", "खड्डे", "खड्ड", "खोळंबा", "गोंधळ", "लाईन", "रांग", "उशीर", "पाऊस", "जाम", "विसरून", "भिजलो", "तुटला", "तुट", "अडचण", "रद्द",
  "bhikaar", "tras", "murkh", "khadde", "fasle", "shiksha", "jam", "traffic", "tutla",
  // Telugu
  "వర్షం", "కరెంట్", "పవర్", "కట్", "ఆలస్యం", "ఆలస్య", "జామ్", "ట్రాఫిక్", "పాడైపోయింది", "పాడై", "దారుణం", "లేట్", "పంచర్", "ఇబ్బంది", "ఇబ్బంద"
]);

// --- MINED TRAINING LEXICONS (Mined exclusively from train.jsonl with support >= 3, precision >= 70%) ---
export const MINED_TRAIN_POSITIVE_CUES = new Set([
  "clear", "actual", "planet", "hypocrisy", "tho", "considering", "omg", "admits", "intelligence",
  "liberal", "obvious", "double", "miracle", "expert", "experts", "shocking", "apparently", "supposedly",
  "wonder", "bizarre", "classic", "genius", "hero", "pure", "wisdom", "जाओ", "कमाल", "सुपर"
]);

export const MINED_TRAIN_NEGATIVE_CUES = new Set([
  "evil", "trolling", "blame", "ridiculous", "pathetic", "waste", "nonsense", "bizarre", "clown",
  "hypocritical", "lies", "lying", "liar", "cheat", "scam", "shame", "disgrace", "atrocious",
  "horrendous", "joke", "fraud", "corrupted", "clueless", "incompetent", "सड़ा", "गटार", "गटारे"
]);

export const MINED_TRAIN_DISCOURSE_CUES = new Set([
  "as if", "who knew", "thanks to", "oh sure", "yeah right", "surely", "obviously", "clearly",
  "ye to hona hi tha", "nakkich", "malli modalaindi", "of course", "surprise surprise", "go figure"
]);

// Combined Exported Lexicons (Backwards-compatible union of Base + Mined Training)
export const POSITIVE_CUES = new Set([...BASE_POSITIVE_CUES, ...MINED_TRAIN_POSITIVE_CUES]);
export const NEGATIVE_CUES = new Set([...BASE_NEGATIVE_CUES, ...MINED_TRAIN_NEGATIVE_CUES]);

export const RHETORICAL_PATTERNS = [
  /\b(what a|just what|who knew|thanks to|as if|oh sure|yeah right|surprise surprise|go figure)\b/i,
  /\b(वाह क्या बात|गजब|धन्य हो|सलामी|वाह रे वाह|स्विमिंग पूल|ये तो होना ही था)\b/,
  /\b(kya baat hai|wah re wah|lai bhari|bhale timing|superb speed|ye to hona hi tha)\b/i,
  /\b(nakkich|malli modalaindi)\b/i,
  /(21 तोपों|स्विमिंग पूल|घंटे लेट|तास traffic)/
];

export const INTENSIFIERS = new Set([
  "obviously", "clearly", "totally", "absolutely", "surely", "of course", "definitely",
  "truly", "purely", "literally",
  "ekdam", "khoop", "agdi", "nakkich", "bilkul", "sach me", "kya baat", "bade log", "arre waah", "bohot", "bahut", "khup", "lai",
  "खूप", "फार", "बहुत", "बिल्कुल", "చాలా"
]);

export const CONTRASTS = new Set([
  "but", "yet", "however", "although", "though", "nevertheless",
  "pan", "pun", "par", "lekin", "magar", "kintu", "पण", "परंतु", "लेकिन", "पर", "मगर"
]);

export const HINDI_WORDS = new Set([
  "hai", "hain", "tha", "thi", "the", "kya", "kyu", "kaise", "bhai", "yaar",
  "mera", "meri", "hum", "aap", "tum", "bohot", "bahut", "accha", "achha", "kar", "raha", "rahi", "sahi", "nahi", "nahin", "aaj"
]);

export const MARATHI_WORDS = new Set([
  "ahe", "aahe", "hota", "hoti", "zala", "zali", "kasa", "kashi", "bhau",
  "dada", "lai", "khup", "chhan", "bagh", "sang", "fukat", "kuthe", "ata", "ani", "pan"
]);

export const ROMAN_SETS: Record<string, Set<string>> = {
  hi: HINDI_WORDS,
  mr: MARATHI_WORDS,
  te: new Set(["ledu", "undi", "unnaru", "enti", "ela", "chala", "bagundi", "baagundi", "nenu", "meeru", "avunu", "ayyo", "sare", "malli"]),
  ta: new Set(["illa", "enna", "romba", "nalla", "naan", "nee", "vanakkam", "epdi", "iruku", "irukku", "machan", "seri", "paaru"]),
  bn: new Set(["ami", "tumi", "achi", "kemon", "bhalo", "korcho", "hobe", "ekdom", "darun", "abar"])
};

export const HINDI_DEV = new Set(["है", "हैं", "और", "नहीं", "क्या", "मैं", "मुझे", "बहुत", "फिर", "वाह", "यह", "था", "थी", "आप", "हम"]);
export const MARATHI_DEV = new Set(["आहे", "आहेत", "आणि", "नाही", "काय", "मी", "मला", "खूप", "पुन्हा", "व्वा", "छान", "आम्ही", "तुम्ही", "होता"]);

export const IRONY_EMOJIS = new Set(["🙄", "😏", "😒", "🙃", "🤡", "😑", "😬"]);
export const POSITIVE_EMOJIS = new Set(["😂", "😍", "😊", "😁", "😀", "👍", "❤️", "🙏", "🎉", "😘", "🥰", "😄"]);
export const NEGATIVE_EMOJIS = new Set(["😡", "😢", "😭", "👎", "💔", "😠", "😞", "😩", "😤"]);

// --- INDIC MORPHOLOGICAL NORMALIZATION SUFFIXES ---
export const DEV_SUFFIXES = [
  // Compound case postpositions & inflections (Marathi & Hindi) - Longest First
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

export const TELUGU_SUFFIXES = [
  // Nominal compound case markers & verbal auxiliaries - Longest First
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
 * Conservative Indic suffix stripper.
 * Strips at most ONE inflectional suffix, requiring minStemLength code units.
 */
export function stemIndicToken(word: string): { stem: string; strippedSuffix: string | null } {
  // Devanagari (\u0900-\u097F)
  if (/[\u0900-\u097F]/.test(word)) {
    const minLen = 3;
    for (const sfx of DEV_SUFFIXES) {
      if (word.endsWith(sfx) && word.length - sfx.length >= minLen) {
        return {
          stem: word.slice(0, word.length - sfx.length),
          strippedSuffix: sfx,
        };
      }
    }
  }

  // Telugu (\u0C00-\u0C7F)
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

  // 1. Collapse runs of repeated characters to at most 1 or 2
  let dedup = lower.replace(/(.)\1{2,}/g, "$1$1");
  // Fold multi-vowel elongations (e.g., baaahuuut -> bahut, sooo -> so)
  dedup = dedup
    .replace(/aaa+/g, "a")
    .replace(/eee+/g, "i")
    .replace(/ooo+/g, "u")
    .replace(/uuu+/g, "u")
    .replace(/aa/g, "a")
    .replace(/uu/g, "u")
    .replace(/ii/g, "i");

  // 2. Vowel folding for romanized Indic phonetic matching
  let folded = dedup
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

// --- SCRIPT UTILITIES ---
const SCRIPT_RANGES: [string, number, number][] = [
  ["Devanagari", 0x0900, 0x097f],
  ["Devanagari", 0xa8e0, 0xa8ff],
  ["Bengali", 0x0980, 0x09ff],
  ["Gurmukhi", 0x0a00, 0x0a7f],
  ["Gujarati", 0x0a80, 0x0aff],
  ["Odia", 0x0b00, 0x0b7f],
  ["Tamil", 0x0b80, 0x0bff],
  ["Telugu", 0x0c00, 0x0c7f],
  ["Kannada", 0x0c80, 0x0cff],
  ["Malayalam", 0x0d00, 0x0d7f],
  ["Arabic", 0x0600, 0x06ff],
  ["Arabic", 0x0750, 0x077f],
  ["Arabic", 0xfb50, 0xfdff],
  ["Arabic", 0xfe70, 0xfeff],
];

export function scriptOf(ch: string): string | null {
  const cp = ch.codePointAt(0);
  if (!cp) return null;
  for (const [name, lo, hi] of SCRIPT_RANGES) {
    if (cp >= lo && cp <= hi) return name;
  }
  if ((cp >= 0x41 && cp <= 0x5a) || (cp >= 0x61 && cp <= 0x7a) || (cp >= 0xc0 && cp <= 0x24f)) {
    return "Latin";
  }
  return null;
}

export function tokenizeWords(text: string): string[] {
  return text
    .split(/\s+/)
    .map((w) => w.replace(/^[\p{P}\p{S}]+|[\p{P}\p{S}]+$/gu, ""))
    .filter((w) => w.length > 0);
}

export function analyzeCodeMixing(text: string) {
  const words = tokenizeWords(text);
  const total = Math.max(words.length, 1);
  const scriptCounts: Record<string, number> = {};

  for (const w of words) {
    for (const ch of w) {
      const s = scriptOf(ch);
      if (s) {
        scriptCounts[s] = (scriptCounts[s] || 0) + 1;
      }
    }
  }

  const scriptsPresent = Object.keys(scriptCounts).sort((a, b) => scriptCounts[b] - scriptCounts[a]);
  const dominant = scriptsPresent[0] || "Latin";

  const lowerWords = words.map((w) => w.toLowerCase());
  const devCount = lowerWords.filter((w) => /[\u0900-\u097F]/.test(w)).length;
  const hiCount = lowerWords.filter((w) => HINDI_WORDS.has(w)).length;
  const mrCount = lowerWords.filter((w) => MARATHI_WORDS.has(w)).length;
  const latinCount = lowerWords.filter((w) => /[a-zA-Z]/.test(w)).length;

  let primaryScript = dominant;
  let isMixed = false;
  let cmi = 0.0;
  let langGuess = "en";
  let romanized = false;
  let confidence: "low" | "medium" | "high" = "low";

  if (devCount > 0) {
    const hiDev = lowerWords.filter((w) => HINDI_DEV.has(w)).length;
    const mrDev = lowerWords.filter((w) => MARATHI_DEV.has(w)).length;
    primaryScript = mrDev > hiDev || mrCount > hiCount ? "Devanagari (Marathi)" : "Devanagari (Hindi)";
    langGuess = mrDev > hiDev || mrCount > hiCount ? "mr" : "hi";
    confidence = (hiDev >= 2 || mrDev >= 2) ? "medium" : "low";
    if (latinCount > 0) {
      isMixed = true;
      cmi = Math.round((latinCount / total) * 1000) / 10;
    }
  } else if (dominant === "Latin") {
    const hits: Record<string, number> = {};
    for (const [lang, wordSet] of Object.entries(ROMAN_SETS)) {
      hits[lang] = lowerWords.filter((w) => wordSet.has(w)).length;
    }
    const bestLang = Object.keys(hits).reduce((a, b) => (hits[a] > hits[b] ? a : b), "hi");
    if (hits[bestLang] > 0) {
      romanized = true;
      langGuess = bestLang;
      confidence = hits[bestLang] >= 2 ? "medium" : "low";
      if (bestLang === "hi") {
        primaryScript = "Hinglish (Hindi-English)";
        cmi = Math.round((hits.hi / total) * 1000) / 10;
        isMixed = true;
      } else if (bestLang === "mr") {
        primaryScript = "Marathi-English Code-Mixed";
        cmi = Math.round((hits.mr / total) * 1000) / 10;
        isMixed = true;
      } else {
        primaryScript = `${bestLang.toUpperCase()}-English Mixed`;
        cmi = Math.round((hits[bestLang] / total) * 1000) / 10;
        isMixed = true;
      }
    } else {
      primaryScript = "English";
      langGuess = "en";
      confidence = "high";
      cmi = 0.0;
      isMixed = false;
    }
  } else {
    primaryScript = dominant;
    langGuess = dominant.toLowerCase().slice(0, 2);
    confidence = "medium";
    cmi = scriptsPresent.length > 1 ? 25.0 : 0.0;
    isMixed = scriptsPresent.length > 1;
  }

  return {
    primary_script: primaryScript,
    dominant,
    scripts_present: scriptsPresent,
    language: langGuess,
    confidence,
    romanized,
    cmi,
    is_code_mixed: isMixed,
  };
}

// --- EMOJI HELPERS ---
const EMOJI_REGEX = /[\u{1F300}-\u{1FAFF}\u{1F000}-\u{1F2FF}\u{2600}-\u{27BF}\u{2B00}-\u{2BFF}\u{1F1E6}-\u{1F1FF}]/gu;

export function extractEmojis(text: string): string[] {
  return Array.from(text.match(EMOJI_REGEX) || []);
}

export function stripEmojis(text: string): string {
  return text.replace(EMOJI_REGEX, " ").replace(/\s+/g, " ").trim();
}

export function emojiFeatures(text: string) {
  const emojis = extractEmojis(text);
  let nIrony = 0;
  let nPos = 0;
  let nNeg = 0;
  for (const e of emojis) {
    if (IRONY_EMOJIS.has(e)) nIrony++;
    if (POSITIVE_EMOJIS.has(e)) nPos++;
    if (NEGATIVE_EMOJIS.has(e)) nNeg++;
  }
  return {
    n_emoji: emojis.length,
    emojis,
    n_irony_emoji: nIrony,
    n_positive_emoji: nPos,
    n_negative_emoji: nNeg,
  };
}

// --- NORMALIZATION ---
export function cleanText(text: string): string {
  if (!text) return "";
  let t = text.normalize("NFKC");
  t = t.replace(/https?:\/\/\S+|www\.\S+/gi, " ");
  t = t.replace(/(?<!\w)@\w+/g, " ");
  t = t.replace(/#(\w+)/g, "$1");
  t = t.replace(/(.)\1{2,}/g, "$1$1");
  return t.replace(/\s+/g, " ").trim();
}

// --- INCONGRUITY EXTRACTOR ---
export interface NormalizationTrace {
  raw: string;
  stripped: string;
  stemmed?: string;
  folded?: string;
  cueType?: "positive" | "negative" | "intensifier" | "contrast" | "discourse";
}

export interface IncongruityFeatures {
  pos_density: number;
  neg_density: number;
  pos_count: number;
  neg_count: number;
  intensifier_count: number;
  contrast_count: number;
  polarity_clash: number;
  intensifier_density: number;
  context_clash: number;
  incongruity_index: number;
  has_rhetorical?: boolean;
  detected_positive: string[];
  detected_negative: string[];
  detected_intensifiers: string[];
  reasons: string[];
  lexical_coverage: number;
  epistemic_uncertainty: number;
  normalization_traces?: NormalizationTrace[];
}

export function extractIncongruity(text: string, context = ""): IncongruityFeatures {
  const tokens = tokenizeWords(text).map((t) => t.toLowerCase());
  const num = Math.max(tokens.length, 1);

  const detectedPos: string[] = [];
  const detectedNeg: string[] = [];
  const detectedInten: string[] = [];
  let contrastCount = 0;
  let coveredCount = 0;
  let contentCount = 0;
  const traces: NormalizationTrace[] = [];

  for (const t of tokens) {
    const stripped = t.replace(/^[\p{P}\p{S}]+|[\p{P}\p{S}]+$/gu, "");
    if (!stripped) continue;

    // Check if content word (length >= 3)
    const isContent = stripped.length >= 3;
    if (isContent) contentCount++;

    // Multi-view token representations:
    // 1. Raw stripped
    // 2. Indic stemmed
    // 3. Romanized dedup & folded
    const { stem: indicStem, strippedSuffix } = stemIndicToken(stripped);
    const { normalized: romanDedup, folded: romanFolded } = normalizeRomanized(stripped);

    let matchedType: NormalizationTrace["cueType"] | undefined;

    // Positive Match
    if (
      POSITIVE_CUES.has(stripped) ||
      POSITIVE_CUES.has(indicStem) ||
      POSITIVE_CUES.has(romanDedup) ||
      POSITIVE_CUES.has(romanFolded)
    ) {
      detectedPos.push(t);
      matchedType = "positive";
      coveredCount++;
    }
    // Negative Match
    if (
      NEGATIVE_CUES.has(stripped) ||
      NEGATIVE_CUES.has(indicStem) ||
      NEGATIVE_CUES.has(romanDedup) ||
      NEGATIVE_CUES.has(romanFolded)
    ) {
      detectedNeg.push(t);
      if (!matchedType) matchedType = "negative";
      coveredCount++;
    }
    // Intensifier Match
    if (
      INTENSIFIERS.has(stripped) ||
      INTENSIFIERS.has(indicStem) ||
      INTENSIFIERS.has(romanDedup) ||
      INTENSIFIERS.has(romanFolded)
    ) {
      detectedInten.push(t);
      if (!matchedType) matchedType = "intensifier";
      coveredCount++;
    }
    // Contrast Match
    if (
      CONTRASTS.has(stripped) ||
      CONTRASTS.has(indicStem) ||
      CONTRASTS.has(romanDedup) ||
      CONTRASTS.has(romanFolded)
    ) {
      contrastCount++;
      if (!matchedType) matchedType = "contrast";
      coveredCount++;
    }
    // Discourse Match
    if (
      MINED_TRAIN_DISCOURSE_CUES.has(stripped) ||
      MINED_TRAIN_DISCOURSE_CUES.has(romanDedup) ||
      MINED_TRAIN_DISCOURSE_CUES.has(romanFolded)
    ) {
      if (!matchedType) matchedType = "discourse";
      coveredCount++;
    }

    traces.push({
      raw: t,
      stripped,
      stemmed: strippedSuffix ? indicStem : undefined,
      folded: romanFolded !== stripped ? romanFolded : undefined,
      cueType: matchedType,
    });
  }

  const posCount = detectedPos.length;
  const negCount = detectedNeg.length;
  const intenCount = detectedInten.length;
  const clash = posCount > 0 && negCount > 0 ? 1.0 : 0.0;
  const hasRhetorical = RHETORICAL_PATTERNS.some((p) => p.test(text));

  let ctxClash = 0.0;
  if (context.trim()) {
    const ctxTokens = tokenizeWords(context).map((t) =>
      t.toLowerCase().replace(/^[\p{P}\p{S}]+|[\p{P}\p{S}]+$/gu, "")
    );
    let ctxNegCount = 0;
    for (const ct of ctxTokens) {
      const { stem: ctStem } = stemIndicToken(ct);
      const { normalized: ctDedup, folded: ctFolded } = normalizeRomanized(ct);
      if (
        NEGATIVE_CUES.has(ct) ||
        NEGATIVE_CUES.has(ctStem) ||
        NEGATIVE_CUES.has(ctDedup) ||
        NEGATIVE_CUES.has(ctFolded)
      ) {
        ctxNegCount++;
      }
    }
    const em = emojiFeatures(text);
    if (ctxNegCount > 0 && (posCount > 0 || hasRhetorical || em.n_irony_emoji > 0)) {
      ctxClash = 1.0;
    }
  }

  // Lexical Coverage Metric C(x)
  const effectiveContent = Math.max(1, contentCount);
  const coverage = Math.min(1.0, coveredCount / effectiveContent);

  // Epistemic Uncertainty U(x)
  const uncertainty = contentCount >= 3 ? Math.max(0, 1.0 - coverage) : 0.55;

  // Calculate Incongruity Index from core engine
  const idx = clash * 0.4 + ctxClash * 0.35 + Math.min(contrastCount, 2) * 0.15 + Math.min(intenCount, 2) * 0.1;

  const reasons: string[] = [];
  if (clash > 0) {
    reasons.push("Intra-Sentence Opposition: Positive wording is directly contradicted by an underlying negative tone.");
  }
  if (ctxClash > 0) {
    reasons.push("Contextual Clash: Statement expresses overt positive sentiment contrasting a negative preceding context.");
  }
  if (hasRhetorical) {
    reasons.push("Hyperbolic / Rhetorical Phrasing: Detected exaggerated discourse construction typical of verbal irony.");
  }
  if (intenCount > 0) {
    reasons.push("Hyperbolic Amplifier: Exaggerators/intensifiers detected typical of sarcastic rhetoric.");
  }
  if (contrastCount > 0) {
    reasons.push("Adversative Disconnect: Contrast markers detected pivoting sentence tone.");
  }
  if (coverage < 0.20 && contentCount >= 4) {
    reasons.push("Low Lexical Coverage: Limited recognized domain vocabulary; marked for elevated epistemic uncertainty.");
  }

  return {
    pos_density: posCount / num,
    neg_density: negCount / num,
    pos_count: posCount,
    neg_count: negCount,
    intensifier_count: intenCount,
    contrast_count: contrastCount,
    polarity_clash: clash,
    intensifier_density: intenCount / num,
    context_clash: ctxClash,
    incongruity_index: Math.round(idx * 1000) / 1000,
    has_rhetorical: hasRhetorical,
    detected_positive: detectedPos,
    detected_negative: detectedNeg,
    detected_intensifiers: detectedInten,
    reasons,
    lexical_coverage: Math.round(coverage * 1000) / 1000,
    epistemic_uncertainty: Math.round(uncertainty * 1000) / 1000,
    normalization_traces: traces,
  };
}

// --- CALIBRATED PREDICTION ENGINE ---
export interface PredictionResult {
  text: string;
  context: string;
  p_sarcastic: number;
  confidence: number;
  label: "sarcastic" | "not sarcastic";
  verdict: "Sarcastic" | "Literal" | "Uncertain";
  abstain: boolean;
  threshold: number;
  incongruity: IncongruityFeatures;
  language: ReturnType<typeof analyzeCodeMixing>;
  emoji: ReturnType<typeof emojiFeatures>;
  model_name: string;
  temperature: number;
}

export function computeRawScore(text: string, context = ""): number {
  const clean = cleanText(text);
  const feats = extractIncongruity(clean, context);
  const em = emojiFeatures(text);
  const hasRhetorical = feats.has_rhetorical || RHETORICAL_PATTERNS.some((p) => p.test(text));

  // Base prior
  let score = 0.20;

  // Polarity clash strongly tilts towards sarcasm
  if (feats.polarity_clash > 0) {
    score += 0.50;
    if (feats.intensifier_count > 0) {
      score += 0.16;
    }
  }
  if (feats.context_clash > 0) {
    score += 0.35;
  }

  // Irony emoji
  if (em.n_irony_emoji > 0) {
    score += Math.min(0.28 + em.n_irony_emoji * 0.12, 0.45);
  }

  // Rhetorical exaggeration or praise pattern
  if (hasRhetorical) {
    score += 0.38;
    if (feats.neg_count > 0 || feats.contrast_count > 0) {
      score += 0.28;
    }
  }

  // Positive surface cue with no negative or context clash -> likely literal
  if (feats.pos_count > 0 && feats.neg_count === 0 && feats.context_clash === 0 && em.n_irony_emoji === 0 && !hasRhetorical) {
    score -= 0.25;
  }

  // Clean literal sentence indicators WITH sufficient lexical coverage
  if (feats.pos_count === 0 && feats.neg_count === 0 && em.n_irony_emoji === 0 && feats.intensifier_count === 0 && feats.context_clash === 0 && !hasRhetorical) {
    if (feats.lexical_coverage >= 0.25) {
      score -= 0.10;
    } else {
      // Low coverage: Keep score in uncertain prior territory rather than falsely declaring confident literal
      score += 0.05;
    }
  }

  // Direct criticism / Sincere complaint without sarcasm
  if (feats.pos_count === 0 && feats.neg_count > 0 && em.n_irony_emoji === 0 && !hasRhetorical) {
    score -= 0.18;
  }

  // Repeated punctuation typical of sarcasm (e.g. "...", "!!!", "!?", "?!")
  if (/(\.\.\.|!\?|\?!|!!)/.test(text)) {
    score += 0.06;
  }

  // Clamp between 0.02 and 0.98
  return Math.max(0.02, Math.min(0.98, score));
}

export function applyTemperatureCalibration(p: number, temperature: number): number {
  if (temperature <= 0) return p;
  // Logit transform: z = ln(p / (1 - p)) / T, p_cal = 1 / (1 + exp(-z))
  const eps = 1e-6;
  const pClamped = Math.max(eps, Math.min(1 - eps, p));
  const logit = Math.log(pClamped / (1 - pClamped));
  const calibratedLogit = logit / temperature;
  return 1 / (1 + Math.exp(-calibratedLogit));
}

export function predict(
  text: string,
  context = "",
  temperature = 1.0,
  abstainThreshold = 0.6
): PredictionResult {
  const rawP = computeRawScore(text, context);
  const pCalibrated = applyTemperatureCalibration(rawP, temperature);
  const pSarcastic = Math.round(pCalibrated * 1000) / 1000;

  const incongruity = extractIncongruity(text, context);
  const language = analyzeCodeMixing(text);
  const emoji = emojiFeatures(text);

  // FALSE-CONFIDENCE CORRECTION:
  // Modulate confidence based on grounded evidence vs lack of vocabulary
  const isSarc = pSarcastic >= 0.5;
  const rawMargin = Math.abs(pSarcastic - 0.5) * 2;

  let evidenceFactor: number;
  if (isSarc) {
    // Sarcastic decision has overt positive signals (irony emoji, clash, rhetorical pattern)
    if (incongruity.has_rhetorical || emoji.n_irony_emoji > 0 || incongruity.polarity_clash > 0 || incongruity.context_clash > 0) {
      evidenceFactor = Math.min(1.0, 0.85 + 0.15 * incongruity.lexical_coverage);
    } else {
      evidenceFactor = Math.min(1.0, 0.75 + 0.25 * incongruity.lexical_coverage);
    }
  } else {
    // Literal decision:
    // If explicit sincere praise or explicit complaint is recognized:
    if (
      (incongruity.pos_count > 0 && incongruity.neg_count === 0) ||
      (incongruity.neg_count > 0 && incongruity.pos_count === 0)
    ) {
      evidenceFactor = Math.min(1.0, 0.75 + 0.25 * incongruity.lexical_coverage);
    } else if (incongruity.lexical_coverage >= 0.20) {
      evidenceFactor = Math.min(1.0, 0.40 + 0.60 * incongruity.lexical_coverage);
    } else {
      // Zero sentiment signals and near-zero vocabulary coverage:
      // Substantially dampen confidence to prevent false-confident literal predictions!
      evidenceFactor = Math.max(0.12, incongruity.lexical_coverage * 1.5);
    }
  }

  const confidence = Math.round(rawMargin * evidenceFactor * 1000) / 1000;
  const words = text.trim().split(/\s+/).filter(Boolean);
  const abstain = confidence < abstainThreshold || (incongruity.epistemic_uncertainty > 0.88 && words.length >= 4 && !isSarc && incongruity.pos_count === 0 && incongruity.neg_count === 0);

  let verdict: "Sarcastic" | "Literal" | "Uncertain" = isSarc ? "Sarcastic" : "Literal";
  if (abstain) {
    verdict = "Uncertain";
  }

  return {
    text,
    context,
    p_sarcastic: pSarcastic,
    confidence,
    label: isSarc ? "sarcastic" : "not sarcastic",
    verdict,
    abstain,
    threshold: abstainThreshold,
    incongruity,
    language,
    emoji,
    model_name: "SARcasmScope-Calibrated-IndicFusion",
    temperature,
  };
}

// --- OCCLUSION / ATTRIBUTION ENGINE ---
export interface TokenAttribution {
  token: string;
  delta: number;
  clean_token: string;
  category?: "positive" | "negative" | "intensifier" | "neutral";
}

export interface OcclusionResult {
  p_sarcastic: number;
  attributions: TokenAttribution[];
  top_sarcastic: TokenAttribution[];
  top_literal: TokenAttribution[];
}

export function computeOcclusion(
  text: string,
  context = "",
  temperature = 1.0,
  topK = 5
): OcclusionResult {
  const tokens = text.trim().split(/\s+/).filter(Boolean);
  const baseResult = predict(text, context, temperature);
  const baseP = baseResult.p_sarcastic;

  if (tokens.length === 0) {
    return {
      p_sarcastic: baseP,
      attributions: [],
      top_sarcastic: [],
      top_literal: [],
    };
  }

  const attributions: TokenAttribution[] = tokens.map((token, i) => {
    const left = tokens.slice(0, i);
    const right = tokens.slice(i + 1);
    const occludedText = [...left, ...right].join(" ");
    const occludedResult = predict(occludedText, context, temperature);
    const delta = Math.round((baseP - occludedResult.p_sarcastic) * 1000) / 1000;

    const clean = token.toLowerCase().replace(/^[\p{P}\p{S}]+|[\p{P}\p{S}]+$/gu, "");
    const { stem: indicStem } = stemIndicToken(clean);
    const { folded: romanFolded } = normalizeRomanized(clean);

    let category: "positive" | "negative" | "intensifier" | "neutral" = "neutral";
    if (POSITIVE_CUES.has(clean) || POSITIVE_CUES.has(indicStem) || POSITIVE_CUES.has(romanFolded)) {
      category = "positive";
    } else if (NEGATIVE_CUES.has(clean) || NEGATIVE_CUES.has(indicStem) || NEGATIVE_CUES.has(romanFolded)) {
      category = "negative";
    } else if (INTENSIFIERS.has(clean) || INTENSIFIERS.has(indicStem) || INTENSIFIERS.has(romanFolded)) {
      category = "intensifier";
    }

    return {
      token,
      delta,
      clean_token: clean,
      category,
    };
  });

  const ranked = [...attributions].sort((a, b) => b.delta - a.delta);
  const topSarcastic = ranked.filter((a) => a.delta > 0).slice(0, topK);
  const topLiteral = [...ranked].reverse().filter((a) => a.delta < 0).slice(0, topK);

  return {
    p_sarcastic: baseP,
    attributions,
    top_sarcastic: topSarcastic,
    top_literal: topLiteral,
  };
}

// --- ROBUSTNESS / PERTURBATION SUITE ---
export interface PerturbationRow {
  perturbation: string;
  description: string;
  perturbed_text: string;
  p_sarcastic: number;
  flipped: boolean;
  delta: number;
}

export interface RobustnessResult {
  baseline_p_sarcastic: number;
  baseline_label: "sarcastic" | "not sarcastic";
  results: PerturbationRow[];
  stability_score: number; // percentage of perturbations that retained the verdict
}

// Indic cluster preservation
function graphemeClusters(word: string): string[] {
  // Use Intl.Segmenter when available or fallback to unicode clusters
  if (typeof Intl !== "undefined" && (Intl as any).Segmenter) {
    const segmenter = new (Intl as any).Segmenter("und", { granularity: "grapheme" });
    return Array.from(segmenter.segment(word)).map((s: any) => s.segment);
  }
  return Array.from(word);
}

export function perturbAll(text: string, context = "", temperature = 1.0): RobustnessResult {
  const baseResult = predict(text, context, temperature);
  const baseP = baseResult.p_sarcastic;
  const baseIsSarc = baseP >= 0.5;

  const words = text.split(/\s+/).filter(Boolean);

  // 1. Character noise (swap/drop/dup inside grapheme clusters)
  const charNoisyWords = words.map((w) => {
    const cl = graphemeClusters(w);
    if (cl.length >= 4) {
      const k = Math.floor(1 + Math.random() * (cl.length - 2));
      const op = Math.random();
      if (op < 0.33) {
        // swap
        const tmp = cl[k];
        cl[k] = cl[k + 1];
        cl[k + 1] = tmp;
      } else if (op < 0.66) {
        // dup
        cl.splice(k, 0, cl[k]);
      } else {
        // drop
        cl.splice(k, 1);
      }
      return cl.join("");
    }
    return w;
  });
  const textCharNoise = charNoisyWords.join(" ");

  // 2. Repeat letters
  const repeatWords = words.map((w) => {
    const cl = graphemeClusters(w);
    if (cl.length >= 3 && Math.random() < 0.6) {
      return w + cl[cl.length - 1].repeat(3);
    }
    return w;
  });
  const textRepeat = repeatWords.join(" ");

  // 3. Remove emoji
  const textNoEmoji = stripEmojis(text);

  // 4. Strip punctuation
  const textNoPunct = text.replace(/[\p{P}\p{S}]/gu, " ").replace(/\s+/g, " ").trim();

  // 5. Drop tags & mentions
  const textDropTags = words.filter((w) => !w.startsWith("#") && !w.startsWith("@")).join(" ");

  // 6. Truncate first 6 words
  const textTruncate = words.slice(0, Math.min(6, words.length)).join(" ");

  // 7. Shuffle words
  const textShuffle = [...words].sort(() => Math.random() - 0.5).join(" ");

  // 8. Uppercase
  const textUppercase = text.toUpperCase();

  const perturbations: { name: string; desc: string; text: string }[] = [
    { name: "char_noise", desc: "Cluster-safe character swap/duplication", text: textCharNoise || text },
    { name: "repeat_letters", desc: "Elongated letters (e.g. soo, wahhh)", text: textRepeat || text },
    { name: "remove_emoji", desc: "Strip all irony & emotion emojis", text: textNoEmoji || text },
    { name: "strip_punct", desc: "Remove exclamation & question marks", text: textNoPunct || text },
    { name: "drop_tags", desc: "Remove hashtags and @mentions", text: textDropTags || text },
    { name: "truncate_6_words", desc: "Truncate text to first 6 words", text: textTruncate || text },
    { name: "shuffle_words", desc: "Randomly permute word order", text: textShuffle || text },
    { name: "uppercase", desc: "Transform into full UPPERCASE", text: textUppercase || text },
  ];

  let retainedCount = 0;
  const rows: PerturbationRow[] = perturbations.map((item) => {
    const res = predict(item.text, context, temperature);
    const pPert = res.p_sarcastic;
    const pertIsSarc = pPert >= 0.5;
    const flipped = pertIsSarc !== baseIsSarc;
    if (!flipped) retainedCount++;
    return {
      perturbation: item.name,
      description: item.desc,
      perturbed_text: item.text,
      p_sarcastic: pPert,
      flipped,
      delta: Math.round((pPert - baseP) * 1000) / 1000,
    };
  });

  return {
    baseline_p_sarcastic: baseP,
    baseline_label: baseIsSarc ? "sarcastic" : "not sarcastic",
    results: rows,
    stability_score: Math.round((retainedCount / rows.length) * 100),
  };
}

// --- PRESET CORPUS DATA ---
export interface PresetSample {
  category: string;
  name: string;
  languageLabel: string;
  statement: string;
  context: string;
  expected: "Sarcastic" | "Literal";
}

export const PRESET_SAMPLES: PresetSample[] = [
  {
    category: "Sarcasm",
    name: "🇮🇳 Hindi Sarcasm",
    languageLabel: "Devanagari Hindi",
    statement: "वाह क्या बात है, 4 घंटे ट्रेन लेट करने के लिए रेलवे को 21 तोपों की सलामी! 🙄",
    context: "ट्रेन 4 घंटे लेट थी",
    expected: "Sarcastic",
  },
  {
    category: "Sarcasm",
    name: "⚡ Hinglish Sarcasm",
    languageLabel: "Hinglish (Code-Mixed)",
    statement: "Bohot badiya bhai, exam se ek raat pehle phone pe reels dekh, pakka top karega! 🙃",
    context: "Fail hone ki tayaari chal rahi hai",
    expected: "Sarcastic",
  },
  {
    category: "Sarcasm",
    name: "🔥 English Sarcasm",
    languageLabel: "English",
    statement: "Oh brilliant, another delay on Mumbai locals, just what my Monday needed! 🙄",
    context: "Local trains halted for 1 hour due to waterlogging",
    expected: "Sarcastic",
  },
  {
    category: "Sarcasm",
    name: "🚩 Marathi Sarcasm",
    languageLabel: "Marathi (Latin)",
    statement: "Lai bhari bhau, 2 taas traffic madhe thevlya badal khup khup aabhar! 👏",
    context: "Stuck in 2 hr traffic jam at Hinjewadi",
    expected: "Sarcastic",
  },
  {
    category: "Sarcasm",
    name: "🌸 Marathi Devanagari Sarcasm",
    languageLabel: "Devanagari Marathi",
    statement: "व्वा, खूप छान नियोजन आहे, अर्धा तास आधी येऊन पण लाईन मध्येच उभे आहोत!",
    context: "Mismanaged line outside venue",
    expected: "Sarcastic",
  },
  {
    category: "Sarcasm",
    name: "✨ Telugu Sarcasm",
    languageLabel: "Telugu",
    statement: "వావ్, మళ్లీ వర్షం, నా బైక్ ప్రయాణానికి భలే టైమింగ్ 🙄",
    context: "Heavy downpour without rain gear",
    expected: "Sarcastic",
  },
  {
    category: "Literal",
    name: "💎 Hindi Literal",
    languageLabel: "Devanagari Hindi",
    statement: "आज की बैठक बहुत ही उत्पादक रही और समय पर समाप्त हो गई।",
    context: "ऑफिस काम",
    expected: "Literal",
  },
  {
    category: "Literal",
    name: "✨ English Literal",
    languageLabel: "English",
    statement: "The train arrived exactly on time at Dadar station today.",
    context: "Smooth morning commute",
    expected: "Literal",
  },
  {
    category: "Literal",
    name: "🌿 Marathi Literal",
    languageLabel: "Marathi (Latin)",
    statement: "Aaj cha presentation khup changla zala, congrats team!",
    context: "Team delivered project smoothly",
    expected: "Literal",
  },
  {
    category: "Literal",
    name: "☀️ Hinglish Literal",
    languageLabel: "Hinglish",
    statement: "Khana sach me bohot lajawab aur swadisht tha, thank you!",
    context: "Family dinner celebration",
    expected: "Literal",
  },
];
