import { useState } from "react";
import {
  POSITIVE_CUES,
  NEGATIVE_CUES,
  INTENSIFIERS,
  CONTRASTS,
} from "../engine/sarcasmEngine";
import { Sliders, BookOpen, Layers, Target, Scale, Database } from "lucide-react";

interface ProtocolViewProps {
  temperature: number;
  setTemperature: (t: number) => void;
  abstainThreshold: number;
  setAbstainThreshold: (a: number) => void;
}

export function ProtocolView({
  temperature,
  setTemperature,
  abstainThreshold,
  setAbstainThreshold,
}: ProtocolViewProps) {
  const [activeSubTab, setActiveSubTab] = useState<"calibration" | "lexicons" | "annotation">("calibration");

  return (
    <div className="space-y-6">
      {/* Sub Tabs */}
      <div className="flex border-b border-white/10 gap-2">
        <button
          onClick={() => setActiveSubTab("calibration")}
          className={`pb-3 px-3 text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5 transition-colors cursor-pointer border-b-2 ${
            activeSubTab === "calibration"
              ? "border-indigo-500 text-indigo-300"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <Sliders className="w-3.5 h-3.5" />
          Calibration & Selective Abstention
        </button>

        <button
          onClick={() => setActiveSubTab("lexicons")}
          className={`pb-3 px-3 text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5 transition-colors cursor-pointer border-b-2 ${
            activeSubTab === "lexicons"
              ? "border-indigo-500 text-indigo-300"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <Database className="w-3.5 h-3.5" />
          Multilingual Lexicons
        </button>

        <button
          onClick={() => setActiveSubTab("annotation")}
          className={`pb-3 px-3 text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5 transition-colors cursor-pointer border-b-2 ${
            activeSubTab === "annotation"
              ? "border-indigo-500 text-indigo-300"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <BookOpen className="w-3.5 h-3.5" />
          Annotation Guidelines & Protocol
        </button>
      </div>

      {activeSubTab === "calibration" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Temperature Scaling Slider */}
          <div className="bg-slate-900/70 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-base font-semibold text-slate-100 flex items-center gap-2">
                  <Scale className="w-4 h-4 text-indigo-400" />
                  Temperature Scaling (T)
                </h4>
                <p className="text-xs text-slate-400 mt-0.5">
                  Controls post-hoc calibration of probability logits via Platt / temperature scaling.
                </p>
              </div>
              <span className="text-lg font-bold font-mono text-indigo-300 px-3 py-1 rounded-lg bg-indigo-950/60 border border-indigo-800/40">
                {temperature.toFixed(2)}
              </span>
            </div>

            <div className="space-y-2">
              <input
                type="range"
                min="0.2"
                max="2.5"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full accent-indigo-500 cursor-pointer"
              />
              <div className="flex justify-between text-[11px] text-slate-500 font-mono">
                <span>0.2 (Overconfident / Sharp)</span>
                <span>1.0 (Standard)</span>
                <span>2.5 (Conservative / Softened)</span>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 text-xs text-slate-300 space-y-2 font-mono">
              <div className="text-indigo-400 font-bold">Calibration Formula:</div>
              <div>z = logit(p) / T = ln(p / (1 - p)) / T</div>
              <div>P_calibrated = σ(z) = 1 / (1 + exp(-z))</div>
            </div>
          </div>

          {/* Abstention Threshold Slider */}
          <div className="bg-slate-900/70 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-base font-semibold text-slate-100 flex items-center gap-2">
                  <Target className="w-4 h-4 text-amber-400" />
                  Selective Abstention Threshold (τ)
                </h4>
                <p className="text-xs text-slate-400 mt-0.5">
                  Threshold below which the model refuses to guess and outputs "Uncertain".
                </p>
              </div>
              <span className="text-lg font-bold font-mono text-amber-300 px-3 py-1 rounded-lg bg-amber-950/60 border border-amber-800/40">
                {abstainThreshold.toFixed(2)}
              </span>
            </div>

            <div className="space-y-2">
              <input
                type="range"
                min="0.0"
                max="0.95"
                step="0.05"
                value={abstainThreshold}
                onChange={(e) => setAbstainThreshold(parseFloat(e.target.value))}
                className="w-full accent-amber-500 cursor-pointer"
              />
              <div className="flex justify-between text-[11px] text-slate-500 font-mono">
                <span>0.0 (Always decide, never abstain)</span>
                <span>0.6 (Default target 90% selective accuracy)</span>
                <span>0.95 (High precision / High abstention)</span>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 text-xs text-slate-300 space-y-2 font-mono">
              <div className="text-amber-400 font-bold">Selective Decision Rule:</div>
              <div>Confidence = 2 × |P_calibrated - 0.5|</div>
              <div>Decision: if Confidence &lt; τ ➔ "Uncertain", else argmax label</div>
            </div>
          </div>
        </div>
      )}

      {activeSubTab === "lexicons" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-slate-900/70 border border-white/10 rounded-2xl p-5 space-y-3">
            <h4 className="text-sm font-semibold text-sky-300 flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-sky-400" />
              Positive Surface Cues ({POSITIVE_CUES.size} entries)
            </h4>
            <p className="text-xs text-slate-400">
              Words indicating surface praise across English, Hindi, Hinglish, and Marathi.
            </p>
            <div className="flex flex-wrap gap-1.5 max-h-56 overflow-y-auto p-2 bg-slate-950/50 rounded-xl border border-slate-800">
              {Array.from(POSITIVE_CUES).map((w) => (
                <span
                  key={w}
                  className="px-2 py-0.5 rounded bg-sky-950/50 border border-sky-800/40 text-sky-200 text-xs"
                >
                  {w}
                </span>
              ))}
            </div>
          </div>

          <div className="bg-slate-900/70 border border-white/10 rounded-2xl p-5 space-y-3">
            <h4 className="text-sm font-semibold text-rose-300 flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-400" />
              Negative / Failure Cues ({NEGATIVE_CUES.size} entries)
            </h4>
            <p className="text-xs text-slate-400">
              Words indicating displeasure, breakdown, delays, and poor outcomes.
            </p>
            <div className="flex flex-wrap gap-1.5 max-h-56 overflow-y-auto p-2 bg-slate-950/50 rounded-xl border border-slate-800">
              {Array.from(NEGATIVE_CUES).map((w) => (
                <span
                  key={w}
                  className="px-2 py-0.5 rounded bg-rose-950/50 border border-rose-800/40 text-rose-200 text-xs"
                >
                  {w}
                </span>
              ))}
            </div>
          </div>

          <div className="bg-slate-900/70 border border-white/10 rounded-2xl p-5 space-y-3">
            <h4 className="text-sm font-semibold text-yellow-300 flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-yellow-400" />
              Hyperbolic Intensifiers ({INTENSIFIERS.size} entries)
            </h4>
            <p className="text-xs text-slate-400">
              Exaggeration markers that amplify sarcastic contrast (e.g. "obviously", "ekdam", "lai bhari").
            </p>
            <div className="flex flex-wrap gap-1.5 max-h-56 overflow-y-auto p-2 bg-slate-950/50 rounded-xl border border-slate-800">
              {Array.from(INTENSIFIERS).map((w) => (
                <span
                  key={w}
                  className="px-2 py-0.5 rounded bg-yellow-950/50 border border-yellow-800/40 text-yellow-200 text-xs"
                >
                  {w}
                </span>
              ))}
            </div>
          </div>

          <div className="bg-slate-900/70 border border-white/10 rounded-2xl p-5 space-y-3">
            <h4 className="text-sm font-semibold text-purple-300 flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-purple-400" />
              Adversative Contrasts ({CONTRASTS.size} entries)
            </h4>
            <p className="text-xs text-slate-400">
              Contrast markers that pivot between overt statement and underlying reality (e.g. "lekin", "pan", "but").
            </p>
            <div className="flex flex-wrap gap-1.5 max-h-56 overflow-y-auto p-2 bg-slate-950/50 rounded-xl border border-slate-800">
              {Array.from(CONTRASTS).map((w) => (
                <span
                  key={w}
                  className="px-2 py-0.5 rounded bg-purple-950/50 border border-purple-800/40 text-purple-200 text-xs"
                >
                  {w}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeSubTab === "annotation" && (
        <div className="bg-slate-900/70 border border-white/10 rounded-2xl p-6 space-y-5 text-sm text-slate-300 leading-relaxed">
          <div className="border-b border-white/10 pb-3">
            <h4 className="text-base font-semibold text-slate-100">
              Annotation Protocol & Theoretical Grounding
            </h4>
            <p className="text-xs text-slate-400 mt-0.5">
              Derived from SarcasmScope research protocols for code-mixed Indian social NLP.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h5 className="font-semibold text-indigo-300 flex items-center gap-1.5 text-xs uppercase tracking-wider">
                1. Context Anchoring Principle
              </h5>
              <p className="text-xs text-slate-300">
                Sarcasm is inherently contextual. When isolated, "Outstanding performance" appears 100% positive. When anchored with context "Crashed 5 minutes after deployment", the semantic polarity flips completely. SarcasmScope explicitly extracts discourse-level context clashes.
              </p>
            </div>

            <div className="space-y-3">
              <h5 className="font-semibold text-indigo-300 flex items-center gap-1.5 text-xs uppercase tracking-wider">
                2. Code-Mixing Index (CMI)
              </h5>
              <p className="text-xs text-slate-300">
                Measures the degree of script and linguistic alternation. Sarcastic expressions in Indian social media heavily code-switch between English and regional mother tongues (e.g., Hinglish, Marathi-English) to achieve comedic or biting effect.
              </p>
            </div>

            <div className="space-y-3">
              <h5 className="font-semibold text-indigo-300 flex items-center gap-1.5 text-xs uppercase tracking-wider">
                3. Abstention vs Hallucination
              </h5>
              <p className="text-xs text-slate-300">
                In safety-critical moderation or sentiment audit systems, a forced binary label with 51% confidence produces high risk. SarcasmScope enforces selective classification using temperature-calibrated thresholds to output "Uncertain".
              </p>
            </div>

            <div className="space-y-3">
              <h5 className="font-semibold text-indigo-300 flex items-center gap-1.5 text-xs uppercase tracking-wider">
                4. Multi-Annotator Agreement
              </h5>
              <p className="text-xs text-slate-300">
                Guidelines specify testing multi-annotator agreement using Cohen's Kappa (&kappa;) and Krippendorff's Alpha (&alpha;) across native speakers familiar with colloquial idioms.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
