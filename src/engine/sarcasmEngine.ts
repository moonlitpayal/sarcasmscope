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

// --- LEXICONS ---
export const POSITIVE_CUES = new Set([
  // English
  "brilliant", "genius", "amazing", "wonderful", "outstanding", "perfect",
  "love", "lovely", "blessed", "greatest", "fantastic", "superb", "flawless", "hero",
  "great", "awesome", "excellent", "magnificent", "masterpiece", "blessing", "pleasure",
  // Hindi Devanagari
  "बढ़िया", "शानदार", "कमाल", "लाजवाब", "वाह", "गजब", "महान", "शाबाश", "सुंदर", "अद्भुत", "धन्य", "बधाई",
  // Hinglish
  "badiya", "bohot badiya", "gazab", "shandar", "kamaal", "lajawab", "waah", "wah", "zabardast", "sahi", "badhiya",
  // Marathi Devanagari & Latin
  "भारी", "छान", "उत्कृष्ट", "मस्त", "शाब्बास", "कौतुक", "bhari", "lai bhari", "mast", "chhan", "lai", "changla"
]);

export const NEGATIVE_CUES = new Set([
  // English
  "terrible", "horrible", "disaster", "fail", "failed", "worst", "idiot",
  "useless", "garbage", "trash", "broken", "nightmare", "clown", "stupid", "pathetic", "ruined",
  "delay", "delayed", "late", "traffic", "stuck", "crash", "crashed", "cancel", "canceled", "mess",
  // Hindi Devanagari
  "बेकार", "घटिया", "तमाशा", "पागल", "फालतू", "झूठ", "धोखा", "गंदा", "कचरा", "बकवास", "मूर्ख", "परेशान",
  // Hinglish
  "bekar", "ghatiya", "tamasha", "pagal", "faltu", "jhoot", "dhoka", "ganda", "bakwas", "kachra", "fas", "late",
  // Marathi Devanagari & Latin
  "भिकार", "घाणेरडा", "त्रास", "खड्डे", "खोळंबा", "bhikaar", "tras", "murkh", "khadde", "fasle", "shiksha"
]);

export const INTENSIFIERS = new Set([
  "obviously", "clearly", "totally", "absolutely", "surely", "of course", "definitely",
  "truly", "purely", "literally",
  "ekdam", "khoop", "agdi", "nakkich", "bilkul", "sach me", "kya baat", "bade log", "arre waah", "bohot", "bahut", "khup", "lai"
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
  detected_positive: string[];
  detected_negative: string[];
  detected_intensifiers: string[];
  reasons: string[];
}

export function extractIncongruity(text: string, context = ""): IncongruityFeatures {
  const tokens = tokenizeWords(text).map((t) => t.toLowerCase());
  const num = Math.max(tokens.length, 1);

  const detectedPos: string[] = [];
  const detectedNeg: string[] = [];
  const detectedInten: string[] = [];
  let contrastCount = 0;

  for (const t of tokens) {
    const stripped = t.replace(/^[^\w\u0900-\u097f]+|[^\w\u0900-\u097f]+$/gu, "");
    if (POSITIVE_CUES.has(stripped)) detectedPos.push(t);
    if (NEGATIVE_CUES.has(stripped)) detectedNeg.push(t);
    if (INTENSIFIERS.has(stripped)) detectedInten.push(t);
    if (CONTRASTS.has(stripped)) contrastCount++;
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
    const ctxNeg = ctxTokens.filter((t) => NEGATIVE_CUES.has(t)).length;
    if (ctxNeg > 0 && posCount > 0) {
      ctxClash = 1.0;
    }
  }

  // Calculate Incongruity Index from core engine
  const idx = clash * 0.4 + ctxClash * 0.35 + Math.min(contrastCount, 2) * 0.15 + Math.min(intenCount, 2) * 0.1;

  const reasons: string[] = [];
  if (clash > 0) {
    reasons.push("Intra-Sentence Opposition: Positive wording is directly contradicted by an underlying negative tone.");
  }
  if (ctxClash > 0) {
    reasons.push("Contextual Clash: Statement expresses overt positive sentiment contrasting a negative preceding context.");
  }
  if (intenCount > 0) {
    reasons.push("Hyperbolic Amplifier: Exaggerators/intensifiers detected typical of sarcastic rhetoric.");
  }
  if (contrastCount > 0) {
    reasons.push("Adversative Disconnect: Contrast markers detected pivoting sentence tone.");
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
    detected_positive: detectedPos,
    detected_negative: detectedNeg,
    detected_intensifiers: detectedInten,
    reasons,
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
  const cm = analyzeCodeMixing(text);

  // Base prior
  let score = 0.22;

  // Incongruity signals
  score += feats.incongruity_index * 0.58;

  // Polarity clash strongly tilts towards sarcasm
  if (feats.polarity_clash > 0) {
    score += 0.25;
  }
  if (feats.context_clash > 0) {
    score += 0.28;
  }

  // Irony emoji
  if (em.n_irony_emoji > 0) {
    score += Math.min(em.n_irony_emoji * 0.26, 0.45);
  }

  // Positive surface cue with no negative or context clash -> likely literal
  if (feats.pos_count > 0 && feats.neg_count === 0 && feats.context_clash === 0) {
    score -= 0.22;
  }

  // Clean literal sentence indicators
  if (feats.pos_count === 0 && feats.neg_count === 0 && em.n_irony_emoji === 0 && feats.intensifier_count === 0) {
    score -= 0.12;
  }

  // Repeated punctuation typical of sarcasm (e.g. "...", "!!!", "!?", "?!")
  if (/(\.\.\.|!\?|\?!|!!)/.test(text)) {
    score += 0.08;
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
  const confidence = Math.round(Math.abs(pSarcastic - 0.5) * 2 * 1000) / 1000;
  const isSarc = pSarcastic >= 0.5;
  const abstain = confidence < abstainThreshold;

  let verdict: "Sarcastic" | "Literal" | "Uncertain" = isSarc ? "Sarcastic" : "Literal";
  if (abstain) {
    verdict = "Uncertain";
  }

  const incongruity = extractIncongruity(text, context);
  const language = analyzeCodeMixing(text);
  const emoji = emojiFeatures(text);

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

    const clean = token.toLowerCase().replace(/^[^\w\u0900-\u097f]+|[^\w\u0900-\u097f]+$/gu, "");
    let category: "positive" | "negative" | "intensifier" | "neutral" = "neutral";
    if (POSITIVE_CUES.has(clean)) category = "positive";
    else if (NEGATIVE_CUES.has(clean)) category = "negative";
    else if (INTENSIFIERS.has(clean)) category = "intensifier";

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
