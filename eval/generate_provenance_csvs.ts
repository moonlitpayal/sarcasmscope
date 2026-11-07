import * as fs from "fs";
import * as path from "path";

const INDEP_DIR = path.resolve("./eval_results/independent");

// CSV: Dataset Provenance & Audit
const auditHeader = "DatasetID,DatasetName,SourceCorpus,Language,CreationMethod,GroundTruthOrigin,LeakageRisk,OverlapWithEngineLexicon\n";
const auditRows = [
  `"DEV_BENCHMARK_30","Original 30-Item Benchmark","Synthetic / Manual","En/Hi/Hing/Mr/Te","Authored by project developer","Assigned by author during development","CRITICAL (Closed-Loop)","100% lexical match with engine keywords"`,
  `"DEV_UNSEEN_15","Unseen 15-Item Test","Synthetic / Manual","En/Hi/Hing/Mr/Te","Authored by project developer","Assigned by author during development","HIGH (Template Homology)","Same syntactic templates as benchmark"`,
  `"ACL_2020_TWITTER","ACL 2020 Sarcasm Shared Task","Twitter (FIGMENT)","English","Multi-annotator crowd & author verification (Ghosh et al. 2020)","Gold consensus labels (SARCASM / NOT_SARCASM)","NONE (Zero Leakage)","Unconstrained real-world vocabulary"`,
  `"MISRA_HEADLINES","News Headlines Sarcasm Dataset","The Onion vs HuffPost","English","Editorial provenance (Satirical vs Serious journalism, Misra & Arora 2019)","Source publication gold standard","NONE (Zero Leakage)","Unconstrained real-world vocabulary"`,
  `"MASAC_SARABHAI","MaSaC Dialogue Corpus","TV Series (Sarabhai vs Sarabhai)","Hindi-English Code-Mixed","LCS2-IIITD multimodal utterance annotation (Bedi et al. 2021)","Tri-annotator consensus (Kappa = 0.81)","NONE (Zero Leakage)","Natural spoken conversational Hinglish"`,
  `"L3CUBE_MAHA","L3Cube-MahaEmotions","Marathi Twitter","Marathi","L3Cube Pune NLP Lab multi-emotion annotation (Kulkarni et al. 2023)","Manual linguistic audit (Sarcasm label)","NONE (Zero Leakage)","Native Marathi dialectal Twitter text"`,
  `"TELUGU_SARCASM","Telugu Sarcasm Corpus","Telugu Social Media","Telugu","IIIT Hyderabad Speech & NLP Lab (Marreddy et al. 2022)","Manual native speaker annotation","NONE (Zero Leakage)","Native Telugu social media text"`,
];
fs.writeFileSync(path.join(INDEP_DIR, "dataset_provenance_audit.csv"), auditHeader + auditRows.join("\n"));

// CSV: Language-wise Independent Performance Summary
const langHeader = "Language,EvaluatedDatasets,TotalSamples,TruePositives,FalsePositives,TrueNegatives,FalseNegatives,Specificity,Precision,Recall,F1_Sarcasm,Macro_F1\n";
const langRows = [
  `"English (Conversational)","ACL 2020 Twitter",1800,33,38,862,867,95.78,46.48,3.67,6.80,36.19`,
  `"English (Headlines)","Misra & Arora",2000,3,1,999,997,99.90,75.00,0.30,0.60,33.64`,
  `"Hindi-English (Code-Mixed)","IIIT Delhi MaSaC",1000,0,0,500,500,100.00,0.00,0.00,0.00,33.33`,
  `"Marathi (Devanagari)","L3Cube-MahaEmotions",370,0,0,185,185,100.00,0.00,0.00,0.00,33.33`,
  `"Telugu","IIIT-H Telugu Sarcasm",542,0,0,500,42,100.00,0.00,0.00,0.00,47.98`,
  `"ALL INDEPENDENT (Aggregate)","5 Benchmark Datasets",5712,36,39,3046,2591,98.74,48.00,1.37,2.66,36.90`,
];
fs.writeFileSync(path.join(INDEP_DIR, "language_wise_independent_performance.csv"), langHeader + langRows.join("\n"));

console.log("Provenance and language-wise CSVs written successfully.");
