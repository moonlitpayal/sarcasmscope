/**
 * SARcasmScope Independent Generalization Evaluation Pipeline
 * 
 * Fetches and evaluates five publicly available, peer-reviewed, independently curated benchmarks:
 * 1. ACL 2020 Sarcasm Shared Task (Twitter) - English conversational with context (Ghosh et al., ACL 2020)
 * 2. News Headlines Dataset for Sarcasm Detection - English headlines (Misra & Arora, 2019)
 * 3. MaSaC Corpus (LCS2-IIITD) - Hindi-English code-mixed dialogue (Bedi et al., IEEE TAFFC 2021)
 * 4. L3Cube-MahaEmotions - Marathi social media tweets (Kulkarni et al., 2023)
 * 5. Telugu Sarcasm Corpus - Telugu social media (Marreddy et al., 2022)
 */

import { predict } from "../src/engine/sarcasmEngine";
import * as fs from "fs";
import * as path from "path";

const INDEP_DIR = path.resolve("./eval_results/independent");
if (!fs.existsSync(INDEP_DIR)) {
  fs.mkdirSync(INDEP_DIR, { recursive: true });
}

interface EvalSample {
  id: string;
  source: string;
  lang: string;
  domain: string;
  hasContext: boolean;
  context: string;
  text: string;
  groundTruth: 0 | 1;
}

interface EvalResult {
  datasetName: string;
  domain: string;
  language: string;
  hasContext: boolean;
  totalSamples: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  macroF1: number;
  weightedF1: number;
  accCiLower: number;
  accCiUpper: number;
  avgLatencyMs: number;
}

function calculateDetailedMetrics(
  yTrue: number[],
  yPred: number[],
  datasetName: string,
  domain: string,
  language: string,
  hasContext: boolean,
  latencies: number[]
): EvalResult {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === 1 && yPred[i] === 1) tp++;
    else if (yTrue[i] === 0 && yPred[i] === 1) fp++;
    else if (yTrue[i] === 0 && yPred[i] === 0) tn++;
    else if (yTrue[i] === 1 && yPred[i] === 0) fn++;
  }

  const n = yTrue.length;
  const n1 = yTrue.filter((y) => y === 1).length;
  const n0 = yTrue.filter((y) => y === 0).length;

  const accuracy = n > 0 ? (tp + tn) / n : 0;
  const precision1 = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall1 = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1_1 = precision1 + recall1 > 0 ? (2 * precision1 * recall1) / (precision1 + recall1) : 0;

  const precision0 = tn + fn > 0 ? tn / (tn + fn) : 0;
  const recall0 = tn + fp > 0 ? tn / (tn + fp) : 0;
  const f1_0 = precision0 + recall0 > 0 ? (2 * precision0 * recall0) / (precision0 + recall0) : 0;

  const macroF1 = (f1_1 + f1_0) / 2;
  const weightedF1 = n > 0 ? (f1_1 * n1 + f1_0 * n0) / n : 0;

  // 95% Wilson Score Confidence Interval for Accuracy
  const z = 1.96;
  const denominator = 1 + (z * z) / n;
  const centerAdjustedProbability = accuracy + (z * z) / (2 * n);
  const adjustedStandardDeviation = Math.sqrt((accuracy * (1 - accuracy) + (z * z) / (4 * n)) / n);
  const accCiLower = Math.max(0, (centerAdjustedProbability - z * adjustedStandardDeviation) / denominator) * 100;
  const accCiUpper = Math.min(1, (centerAdjustedProbability + z * adjustedStandardDeviation) / denominator) * 100;

  const avgLatencyMs = latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : 0;

  return {
    datasetName,
    domain,
    language,
    hasContext,
    totalSamples: n,
    tp,
    fp,
    tn,
    fn,
    accuracy: accuracy * 100,
    precision: precision1 * 100,
    recall: recall1 * 100,
    f1: f1_1 * 100,
    macroF1: macroF1 * 100,
    weightedF1: weightedF1 * 100,
    accCiLower,
    accCiUpper,
    avgLatencyMs,
  };
}

// -----------------------------------------------------------------------------
// DATASET LOADERS
// -----------------------------------------------------------------------------

async function loadSemEvalTwitter(): Promise<EvalSample[]> {
  console.log("Fetching ACL 2020 / SemEval Twitter Test Set...");
  const url = "https://raw.githubusercontent.com/EducationalTestingService/sarcasm/master/twitter/sarcasm_detection_shared_task_twitter_testing.jsonl";
  const res = await fetch(url);
  const text = await res.text();
  const lines = text.trim().split("\n");
  const samples: EvalSample[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    try {
      const obj = JSON.parse(line);
      const isSarc = obj.label === "SARCASM" ? 1 : 0;
      const ctx = Array.isArray(obj.context) ? obj.context.join(" ") : "";
      samples.push({
        id: `TW_${obj.id || i}`,
        source: "ACL 2020 Shared Task (Twitter)",
        lang: "English",
        domain: "Social Media Dialogue",
        hasContext: true,
        context: ctx,
        text: obj.response,
        groundTruth: isSarc as 0 | 1,
      });
    } catch {
      // ignore parse err
    }
  }
  return samples;
}

async function loadMisraHeadlines(limit = 2000): Promise<EvalSample[]> {
  console.log("Fetching Misra & Arora News Headlines Dataset...");
  const url = "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json";
  const res = await fetch(url);
  const text = await res.text();
  const lines = text.trim().split("\n");
  
  const sarcList: EvalSample[] = [];
  const nonSarcList: EvalSample[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    try {
      const obj = JSON.parse(line);
      const isSarc = obj.is_sarcastic === 1 ? 1 : 0;
      const sample: EvalSample = {
        id: `HEADLINE_${i}`,
        source: "Misra & Arora News Headlines",
        lang: "English",
        domain: "News Headlines",
        hasContext: false,
        context: "",
        text: obj.headline,
        groundTruth: isSarc as 0 | 1,
      };
      if (isSarc === 1 && sarcList.length < limit / 2) sarcList.push(sample);
      else if (isSarc === 0 && nonSarcList.length < limit / 2) nonSarcList.push(sample);
      if (sarcList.length >= limit / 2 && nonSarcList.length >= limit / 2) break;
    } catch {
      // ignore
    }
  }
  return [...sarcList, ...nonSarcList];
}

async function loadMaSaCHindi(limit = 1000): Promise<EvalSample[]> {
  console.log("Fetching IIIT Delhi MaSaC Hindi-English Dataset...");
  const url = "https://raw.githubusercontent.com/LCS2-IIITD/Code-mixed-classification/master/data/hindi_sarcasm.txt";
  const res = await fetch(url);
  const text = await res.text();
  const lines = text.trim().split("\n");

  const sarcList: EvalSample[] = [];
  const nonSarcList: EvalSample[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const parts = line.split("\t");
    if (parts.length < 2) continue;
    const utterance = parts[0].trim();
    const labelStr = parts[1].trim();
    const isSarc = labelStr === "1" ? 1 : 0;

    if (utterance.length < 5) continue;

    const sample: EvalSample = {
      id: `MASAC_${i}`,
      source: "MaSaC (Bedi et al. 2021)",
      lang: "Hindi-English (Hinglish)",
      domain: "TV Dialogue (Sarabhai vs Sarabhai)",
      hasContext: false,
      context: "",
      text: utterance,
      groundTruth: isSarc as 0 | 1,
    };

    if (isSarc === 1 && sarcList.length < limit / 2) sarcList.push(sample);
    else if (isSarc === 0 && nonSarcList.length < limit / 2) nonSarcList.push(sample);
    if (sarcList.length >= limit / 2 && nonSarcList.length >= limit / 2) break;
  }
  return [...sarcList, ...nonSarcList];
}

async function loadMarathiMahaEmotions(): Promise<EvalSample[]> {
  console.log("Fetching L3Cube-MahaEmotions Marathi Dataset...");
  const url = "https://raw.githubusercontent.com/l3cube-pune/MarathiNLP/master/L3Cube-MahaEmotions/train.csv";
  const res = await fetch(url);
  const text = await res.text();
  const lines = text.trim().split("\n");

  const sarcList: EvalSample[] = [];
  const nonSarcList: EvalSample[] = [];

  // Parse CSV (Tweet,Label)
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const lastComma = line.lastIndexOf(",");
    if (lastComma === -1) continue;
    const tweet = line.substring(0, lastComma).replace(/^["']|["']$/g, "").trim();
    const label = line.substring(lastComma + 1).trim();

    if (!tweet || tweet.length < 5) continue;

    if (label === "Sarcasm") {
      sarcList.push({
        id: `MR_EMO_${i}`,
        source: "L3Cube-MahaEmotions (Marathi)",
        lang: "Marathi",
        domain: "Twitter Social Media",
        hasContext: false,
        context: "",
        text: tweet,
        groundTruth: 1,
      });
    } else if (["Neutral", "Happiness", "Anger/Disgust"].includes(label) && nonSarcList.length < 200) {
      nonSarcList.push({
        id: `MR_EMO_${i}`,
        source: "L3Cube-MahaEmotions (Marathi)",
        lang: "Marathi",
        domain: "Twitter Social Media",
        hasContext: false,
        context: "",
        text: tweet,
        groundTruth: 0,
      });
    }
  }

  // Also include test.csv
  const testUrl = "https://raw.githubusercontent.com/l3cube-pune/MarathiNLP/master/L3Cube-MahaEmotions/test.csv";
  const testRes = await fetch(testUrl);
  const testText = await testRes.text();
  const testLines = testText.trim().split("\n");
  for (let i = 1; i < testLines.length; i++) {
    const line = testLines[i].trim();
    if (!line) continue;
    const lastComma = line.lastIndexOf(",");
    if (lastComma === -1) continue;
    const tweet = line.substring(0, lastComma).replace(/^["']|["']$/g, "").trim();
    const label = line.substring(lastComma + 1).trim();

    if (label === "Sarcasm") {
      sarcList.push({
        id: `MR_TEST_${i}`,
        source: "L3Cube-MahaEmotions (Marathi)",
        lang: "Marathi",
        domain: "Twitter Social Media",
        hasContext: false,
        context: "",
        text: tweet,
        groundTruth: 1,
      });
    }
  }

  const balancedNonSarc = nonSarcList.slice(0, sarcList.length);
  return [...sarcList, ...balancedNonSarc];
}

async function loadTeluguSarcasm(limit = 1000): Promise<EvalSample[]> {
  console.log("Fetching IIIT Hyderabad Telugu Sarcasm Dataset...");
  const url = "https://huggingface.co/datasets/mounikaiiith/Telugu-Sarcasm/raw/main/Sarcasm_test.csv";
  const res = await fetch(url);
  const text = await res.text();
  const lines = text.trim().split("\n");

  const sarcList: EvalSample[] = [];
  const nonSarcList: EvalSample[] = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const lastComma = line.lastIndexOf(",");
    if (lastComma === -1) continue;
    const sentence = line.substring(0, lastComma).replace(/^[0-9]+,["']?|["']?$/g, "").trim();
    const label = line.substring(lastComma + 1).trim().toLowerCase();

    const isSarc = label === "yes" ? 1 : 0;
    if (sentence.length < 5) continue;

    const sample: EvalSample = {
      id: `TE_IND_${i}`,
      source: "Telugu-Sarcasm (Marreddy et al. 2022)",
      lang: "Telugu",
      domain: "Telugu Social Media",
      hasContext: false,
      context: "",
      text: sentence,
      groundTruth: isSarc as 0 | 1,
    };

    if (isSarc === 1 && sarcList.length < limit / 2) sarcList.push(sample);
    else if (isSarc === 0 && nonSarcList.length < limit / 2) nonSarcList.push(sample);
    if (sarcList.length >= limit / 2 && nonSarcList.length >= limit / 2) break;
  }

  return [...sarcList, ...nonSarcList];
}

// -----------------------------------------------------------------------------
// MAIN EXECUTION ENGINE
// -----------------------------------------------------------------------------
async function run() {
  console.log("================================================================================");
  console.log("SARcasmScope: Independent Multi-Domain & Cross-Dataset Generalization Evaluation");
  console.log("Evaluating the frozen prediction engine against 5 independent published benchmarks");
  console.log("================================================================================\n");

  const benchmarks = [
    { name: "ACL 2020 Twitter Sarcasm", loader: loadSemEvalTwitter },
    { name: "Misra News Headlines", loader: () => loadMisraHeadlines(2000) },
    { name: "MaSaC Code-Mixed Dialogue", loader: () => loadMaSaCHindi(1000) },
    { name: "L3Cube-MahaEmotions Marathi", loader: loadMarathiMahaEmotions },
    { name: "IIIT-H Telugu Sarcasm", loader: () => loadTeluguSarcasm(1000) },
  ];

  const allSummaryResults: EvalResult[] = [];
  const allDetailedPredictions: Array<{
    id: string;
    source: string;
    lang: string;
    text: string;
    groundTruth: number;
    predictedP: number;
    predictedLabel: number;
    correct: boolean;
  }> = [];

  for (const b of benchmarks) {
    try {
      const samples = await b.loader();
      console.log(`Loaded ${samples.length} items for ${b.name}`);

      const yTrue: number[] = [];
      const yPred: number[] = [];
      const latencies: number[] = [];

      for (const s of samples) {
        const t0 = performance.now();
        const pred = predict(s.text, s.context);
        const t1 = performance.now();
        latencies.push(t1 - t0);

        const predLabel = pred.p_sarcastic >= 0.5 ? 1 : 0;
        yTrue.push(s.groundTruth);
        yPred.push(predLabel);

        allDetailedPredictions.push({
          id: s.id,
          source: s.source,
          lang: s.lang,
          text: s.text.replace(/[\r\n\t,]/g, " "),
          groundTruth: s.groundTruth,
          predictedP: pred.p_sarcastic,
          predictedLabel: predLabel,
          correct: predLabel === s.groundTruth,
        });
      }

      const domain = samples[0]?.domain || "Mixed";
      const lang = samples[0]?.lang || "Mixed";
      const hasCtx = samples[0]?.hasContext || false;

      const metrics = calculateDetailedMetrics(yTrue, yPred, b.name, domain, lang, hasCtx, latencies);
      allSummaryResults.push(metrics);

      console.log(`\n--- RESULTS FOR: ${b.name} ---`);
      console.log(`Samples:     ${metrics.totalSamples}`);
      console.log(`Accuracy:    ${metrics.accuracy.toFixed(2)}% (95% CI: [${metrics.accCiLower.toFixed(2)}%, ${metrics.accCiUpper.toFixed(2)}%])`);
      console.log(`Precision:   ${metrics.precision.toFixed(2)}%`);
      console.log(`Recall:      ${metrics.recall.toFixed(2)}%`);
      console.log(`Sarcasm F1:  ${metrics.f1.toFixed(2)}%`);
      console.log(`Macro F1:    ${metrics.macroF1.toFixed(2)}%`);
      console.log(`Confusion:   TP=${metrics.tp}, FP=${metrics.fp}, TN=${metrics.tn}, FN=${metrics.fn}`);
      console.log(`Avg Latency: ${metrics.avgLatencyMs.toFixed(3)} ms`);
      console.log("--------------------------------------------------------------------------------\n");
    } catch (err) {
      console.error(`Failed to evaluate benchmark ${b.name}:`, err);
    }
  }

  // Save CSV 1: Independent Evaluation Summary
  const summaryCsvHeader = "Dataset,Domain,Language,HasContext,N,TP,FP,TN,FN,Accuracy,Precision,Recall,F1_Sarcasm,Macro_F1,Weighted_F1,95CI_Lower,95CI_Upper,AvgLatency_ms\n";
  const summaryCsvRows = allSummaryResults.map((r) =>
    [
      `"${r.datasetName}"`,
      `"${r.domain}"`,
      `"${r.language}"`,
      r.hasContext,
      r.totalSamples,
      r.tp,
      r.fp,
      r.tn,
      r.fn,
      r.accuracy.toFixed(2),
      r.precision.toFixed(2),
      r.recall.toFixed(2),
      r.f1.toFixed(2),
      r.macroF1.toFixed(2),
      r.weightedF1.toFixed(2),
      r.accCiLower.toFixed(2),
      r.accCiUpper.toFixed(2),
      r.avgLatencyMs.toFixed(4),
    ].join(",")
  );
  fs.writeFileSync(path.join(INDEP_DIR, "independent_evaluation_metrics.csv"), summaryCsvHeader + summaryCsvRows.join("\n"));

  // Save CSV 2: Cross Dataset & Domain Analysis
  const crossCsvHeader = "SourceDomain,TargetDomain,SampleCount,MacroF1,Accuracy,Recall,Precision,KeyFailureMode\n";
  const crossRows = [
    `"Demonstration Templates","Conversational Dialogue (Twitter)",1800,${allSummaryResults.find((x) => x.datasetName.includes("Twitter"))?.macroF1.toFixed(2)},${allSummaryResults.find((x) => x.datasetName.includes("Twitter"))?.accuracy.toFixed(2)},${allSummaryResults.find((x) => x.datasetName.includes("Twitter"))?.recall.toFixed(2)},${allSummaryResults.find((x) => x.datasetName.includes("Twitter"))?.precision.toFixed(2)},"Slang, implicit world knowledge, hashtag irony"`,
    `"Demonstration Templates","News Headlines (The Onion)",2000,${allSummaryResults.find((x) => x.datasetName.includes("Headlines"))?.macroF1.toFixed(2)},${allSummaryResults.find((x) => x.datasetName.includes("Headlines"))?.accuracy.toFixed(2)},${allSummaryResults.find((x) => x.datasetName.includes("Headlines"))?.recall.toFixed(2)},${allSummaryResults.find((x) => x.datasetName.includes("Headlines"))?.precision.toFixed(2)},"Absurdity/satire without explicit polarity contrast"`,
    `"Demonstration Templates","TV Code-Mixed Dialogue (MaSaC)",1000,${allSummaryResults.find((x) => x.datasetName.includes("MaSaC"))?.macroF1.toFixed(2)},${allSummaryResults.find((x) => x.datasetName.includes("MaSaC"))?.accuracy.toFixed(2)},${allSummaryResults.find((x) => x.datasetName.includes("MaSaC"))?.recall.toFixed(2)},${allSummaryResults.find((x) => x.datasetName.includes("MaSaC"))?.precision.toFixed(2)},"Acoustic/prosodic sarcasm, conversational turn-taking"`,
    `"Demonstration Templates","Marathi Social Media (MahaEmotions)",${allSummaryResults.find((x) => x.datasetName.includes("Marathi"))?.totalSamples},${allSummaryResults.find((x) => x.datasetName.includes("Marathi"))?.macroF1.toFixed(2)},${allSummaryResults.find((x) => x.datasetName.includes("Marathi"))?.accuracy.toFixed(2)},${allSummaryResults.find((x) => x.datasetName.includes("Marathi"))?.recall.toFixed(2)},${allSummaryResults.find((x) => x.datasetName.includes("Marathi"))?.precision.toFixed(2)},"Complex inflectional morphology, political satire"`,
    `"Demonstration Templates","Telugu Social Media (Marreddy)",1000,${allSummaryResults.find((x) => x.datasetName.includes("Telugu"))?.macroF1.toFixed(2)},${allSummaryResults.find((x) => x.datasetName.includes("Telugu"))?.accuracy.toFixed(2)},${allSummaryResults.find((x) => x.datasetName.includes("Telugu"))?.recall.toFixed(2)},${allSummaryResults.find((x) => x.datasetName.includes("Telugu"))?.precision.toFixed(2)},"Agglutinative suffixes, sandhi, domain lexicon coverage"`,
  ];
  fs.writeFileSync(path.join(INDEP_DIR, "cross_dataset_domain_generalization.csv"), crossCsvHeader + crossRows.join("\n"));

  // Save CSV 3: Sample Predictions (First 500 items for audit)
  const predCsvHeader = "ID,Dataset,Language,GroundTruth,PredictedP,PredictedLabel,Correct,Text\n";
  const predCsvRows = allDetailedPredictions.slice(0, 1000).map((p) =>
    [
      p.id,
      `"${p.source}"`,
      `"${p.lang}"`,
      p.groundTruth,
      p.predictedP.toFixed(3),
      p.predictedLabel,
      p.correct ? 1 : 0,
      `"${p.text.substring(0, 120)}"`,
    ].join(",")
  );
  fs.writeFileSync(path.join(INDEP_DIR, "independent_sample_predictions.csv"), predCsvHeader + predCsvRows.join("\n"));

  console.log("Evaluation complete. All results saved to eval_results/independent/");
}

run();
