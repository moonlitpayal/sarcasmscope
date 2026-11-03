import { useState } from "react";
import {
  predict,
  computeOcclusion,
  PRESET_SAMPLES,
  PredictionResult,
  OcclusionResult,
  PresetSample,
} from "../engine/sarcasmEngine";
import {
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  ArrowRight,
  Info,
  Maximize2,
  RefreshCw,
} from "lucide-react";

interface StudioViewProps {
  temperature: number;
  abstainThreshold: number;
  onSelectSampleForRobustness?: (text: string, context: string) => void;
}

export function StudioView({
  temperature,
  abstainThreshold,
  onSelectSampleForRobustness,
}: StudioViewProps) {
  const [statement, setStatement] = useState(
    "वाह क्या बात है, 4 घंटे ट्रेन लेट करने के लिए रेलवे को 21 तोपों की सलामी! 🙄"
  );
  const [context, setContext] = useState("ट्रेन 4 घंटे लेट थी");
  const [selectedPreset, setSelectedPreset] = useState<string>("🇮🇳 Hindi Sarcasm");
  const [result, setResult] = useState<PredictionResult | null>(() =>
    predict(
      "वाह क्या बात है, 4 घंटे ट्रेन लेट करने के लिए रेलवे को 21 तोपों की सलामी! 🙄",
      "ट्रेन 4 घंटे लेट थी",
      temperature,
      abstainThreshold
    )
  );
  const [occlusion, setOcclusion] = useState<OcclusionResult | null>(() =>
    computeOcclusion(
      "वाह क्या बात है, 4 घंटे ट्रेन लेट करने के लिए रेलवे को 21 तोपों की सलामी! 🙄",
      "ट्रेन 4 घंटे लेट थी",
      temperature
    )
  );
  const [hoveredToken, setHoveredToken] = useState<{
    token: string;
    delta: number;
    category?: string;
  } | null>(null);

  const handleAnalyze = (newText = statement, newContext = context) => {
    if (!newText.trim()) return;
    const res = predict(newText, newContext, temperature, abstainThreshold);
    const occ = computeOcclusion(newText, newContext, temperature);
    setResult(res);
    setOcclusion(occ);
  };

  const loadPreset = (preset: PresetSample) => {
    setSelectedPreset(preset.name);
    setStatement(preset.statement);
    setContext(preset.context);
    handleAnalyze(preset.statement, preset.context);
  };

  const maxAbsDelta = occlusion?.attributions.length
    ? Math.max(0.001, ...occlusion.attributions.map((a) => Math.abs(a.delta)))
    : 0.1;

  return (
    <div className="space-y-6">
      {/* Quick Preset Selector Chips */}
      <div className="flex flex-wrap items-center gap-2 pb-1 border-b border-white/5">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider mr-2">
          Preset Samples:
        </span>
        {PRESET_SAMPLES.map((p) => {
          const isSelected = selectedPreset === p.name;
          return (
            <button
              key={p.name}
              onClick={() => loadPreset(p)}
              className={`text-xs px-3 py-1.5 rounded-lg border transition-all cursor-pointer flex items-center gap-1.5 ${
                isSelected
                  ? "bg-indigo-600/25 border-indigo-500/60 text-indigo-200 shadow-sm shadow-indigo-500/20"
                  : "bg-slate-900/60 hover:bg-slate-800/80 border-slate-800 text-slate-300 hover:text-white"
              }`}
            >
              <span>{p.name}</span>
              <span className="text-[10px] px-1 rounded bg-white/10 text-slate-400">
                {p.expected}
              </span>
            </button>
          );
        })}
      </div>

      {/* Main 2-Column Studio Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Input Panel */}
        <div className="bg-slate-900/70 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-xl space-y-4 transition-all hover:border-indigo-500/30">
          <div className="flex items-center justify-between">
            <label className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              Target Statement
            </label>
            <span className="text-xs text-slate-500 font-mono">
              {statement.length} chars
            </span>
          </div>

          <textarea
            value={statement}
            onChange={(e) => setStatement(e.target.value)}
            rows={4}
            placeholder="Type or paste statement in Hindi, Hinglish, Marathi, English, Telugu, etc. (Emoji welcome)"
            className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-3.5 text-slate-100 text-base placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/50 resize-y leading-relaxed transition-colors"
          />

          <div className="space-y-1.5">
            <label className="text-sm font-semibold text-slate-200 flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Info className="w-4 h-4 text-slate-400" />
                Conversational Context
                <span className="text-xs font-normal text-slate-500">
                  (Optional preceding discourse)
                </span>
              </span>
            </label>
            <input
              type="text"
              value={context}
              onChange={(e) => setContext(e.target.value)}
              placeholder="e.g. Flight was delayed by 3 hours, Train cancelled, Burnt dinner"
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-3.5 py-2.5 text-slate-100 text-sm placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/50 transition-colors"
            />
          </div>

          <div className="pt-2 flex items-center gap-3">
            <button
              onClick={() => handleAnalyze()}
              disabled={!statement.trim()}
              className="flex-1 py-3 px-5 rounded-xl bg-gradient-to-r from-indigo-600 via-indigo-500 to-indigo-600 hover:from-indigo-500 hover:to-indigo-500 active:scale-[0.99] text-white font-medium text-sm shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-2 transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <RefreshCw className="w-4 h-4" />
              <span>✦ Analyze Sarcasm & Incongruity</span>
            </button>

            {onSelectSampleForRobustness && (
              <button
                onClick={() => onSelectSampleForRobustness(statement, context)}
                title="Send to Robustness Noise Suite"
                className="py-3 px-3.5 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 hover:text-white border border-slate-700 transition-colors cursor-pointer"
              >
                <ArrowRight className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Quick Language & Emoji Metadata Chips */}
          {result && (
            <div className="pt-2 border-t border-white/5 flex flex-wrap gap-1.5 text-xs">
              <span className="px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 text-slate-300">
                Script: <strong className="text-white">{result.language.primary_script}</strong>
              </span>
              <span className="px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 text-slate-300">
                Language: <strong className="text-white">{result.language.language.toUpperCase()}</strong> ({result.language.confidence})
              </span>
              <span className="px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 text-slate-300">
                CMI: <strong className="text-indigo-300">{result.language.cmi}%</strong>
              </span>
              {result.emoji.n_emoji > 0 ? (
                <span className="px-2.5 py-1 rounded-md bg-amber-950/40 border border-amber-800/40 text-amber-200">
                  Emoji: {result.emoji.emojis.join(" ")} ({result.emoji.n_irony_emoji} irony)
                </span>
              ) : (
                <span className="px-2.5 py-1 rounded-md bg-slate-800/50 border border-slate-800 text-slate-400">
                  No emoji
                </span>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Cognitive Verdict & Probability */}
        <div className="bg-slate-900/70 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-xl space-y-5 transition-all hover:border-indigo-500/30">
          <div className="flex items-center justify-between border-b border-white/5 pb-3">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Cognitive Verdict
            </span>
            <span className="text-xs text-slate-500 font-mono">
              Temp: {temperature.toFixed(2)} | τ: {abstainThreshold.toFixed(2)}
            </span>
          </div>

          {result ? (
            <div className="space-y-4">
              {/* Verdict Card */}
              {result.abstain ? (
                <div className="p-5 rounded-xl border bg-amber-950/25 border-amber-500/40 text-amber-200 space-y-1.5 animate-fadeIn">
                  <div className="flex items-center gap-2 text-lg font-bold">
                    <HelpCircle className="w-5 h-5 text-amber-400" />
                    Uncertain: Abstaining from Decision
                  </div>
                  <p className="text-xs text-amber-200/80 leading-relaxed">
                    Model confidence (<strong>{result.confidence.toFixed(3)}</strong>) is below
                    the abstention threshold <strong>{result.threshold.toFixed(2)}</strong>.
                    Rather than hallucinating or guessing, the system flags this for human adjudication.
                  </p>
                </div>
              ) : result.verdict === "Sarcastic" ? (
                <div className="p-5 rounded-xl border bg-gradient-to-br from-rose-950/30 to-rose-900/10 border-rose-500/40 text-rose-200 space-y-1.5 animate-fadeIn">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-xl font-bold text-rose-300">
                      <AlertTriangle className="w-6 h-6 text-rose-400" />
                      Sarcastic Intent Detected
                    </div>
                    <span className="text-xs font-mono font-semibold px-2.5 py-1 rounded-md bg-rose-500/20 text-rose-300 border border-rose-500/30">
                      {(result.p_sarcastic * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-xs text-rose-200/75">
                    High cognitive incongruity detected. Surface positive vocabulary directly contradicts discourse sentiment.
                  </p>
                </div>
              ) : (
                <div className="p-5 rounded-xl border bg-gradient-to-br from-emerald-950/30 to-emerald-900/10 border-emerald-500/40 text-emerald-200 space-y-1.5 animate-fadeIn">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-xl font-bold text-emerald-300">
                      <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                      Literal Statement
                    </div>
                    <span className="text-xs font-mono font-semibold px-2.5 py-1 rounded-md bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                      {((1 - result.p_sarcastic) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-xs text-emerald-200/75">
                    Semantically congruent statement. Surface expressions align naturally with contextual discourse.
                  </p>
                </div>
              )}

              {/* Calibrated Probability Bar */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs text-slate-400 font-mono">
                  <span>Literal (0.0)</span>
                  <span className="font-semibold text-slate-200">
                    P(sarcastic) = {result.p_sarcastic.toFixed(3)}
                  </span>
                  <span>Sarcastic (1.0)</span>
                </div>
                <div className="h-3.5 w-full bg-slate-950 rounded-full overflow-hidden p-0.5 border border-slate-800">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.max(2, result.p_sarcastic * 100)}%`,
                      background:
                        "linear-gradient(90deg, #10b981 0%, #3b82f6 40%, #eab308 70%, #f43f5e 100%)",
                    }}
                  />
                </div>
              </div>

              {/* 3 Metric Cards */}
              <div className="grid grid-cols-3 gap-3 pt-1">
                <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                  <span className="text-[11px] text-slate-400 uppercase font-medium">
                    Sarcasm P
                  </span>
                  <div className="text-lg font-bold text-slate-100 font-mono mt-0.5">
                    {result.p_sarcastic.toFixed(3)}
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                  <span className="text-[11px] text-slate-400 uppercase font-medium">
                    Confidence
                  </span>
                  <div className="text-lg font-bold text-slate-100 font-mono mt-0.5">
                    {result.confidence.toFixed(3)}
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                  <span className="text-[11px] text-slate-400 uppercase font-medium">
                    Incongruity Idx
                  </span>
                  <div className="text-lg font-bold text-indigo-300 font-mono mt-0.5">
                    {result.incongruity.incongruity_index.toFixed(2)}
                  </div>
                </div>
              </div>

              {/* Linguistic Reasoning Explanation */}
              <div className="space-y-2 pt-2 border-t border-white/5">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
                  Linguistic Incongruity Explanation
                </span>
                {result.incongruity.reasons.length > 0 ? (
                  <ul className="space-y-1.5 text-xs text-slate-300">
                    {result.incongruity.reasons.map((r, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="text-indigo-400 mt-0.5">•</span>
                        <span>{r}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-slate-400">
                    Tone and vocabulary are semantically congruent with the context. No intra-sentential contradiction found.
                  </p>
                )}
              </div>
            </div>
          ) : (
            <div className="py-12 text-center text-slate-500 text-sm">
              Enter a statement and click Analyze to view classification.
            </div>
          )}
        </div>
      </div>

      {/* Token Influence / Occlusion Attribution Panel */}
      {occlusion && occlusion.attributions.length > 0 && (
        <div className="bg-slate-900/70 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-xl space-y-4 transition-all hover:border-indigo-500/30">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-white/5 pb-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <span>Why? Token Influence (Leave-One-Token-Out Occlusion)</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                <span className="text-rose-400 font-medium">Rose/Red</span> pushes towards sarcasm,{" "}
                <span className="text-sky-400 font-medium">Sky/Blue</span> pushes towards literal.
                Higher color intensity indicates greater attribution magnitude (Δ).
              </p>
            </div>

            {hoveredToken && (
              <div className="text-xs font-mono bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-300">
                Word: <strong className="text-white">"{hoveredToken.token}"</strong> | Δ:{" "}
                <span
                  className={
                    hoveredToken.delta >= 0 ? "text-rose-400 font-bold" : "text-sky-400 font-bold"
                  }
                >
                  {hoveredToken.delta >= 0 ? `+${hoveredToken.delta.toFixed(3)}` : hoveredToken.delta.toFixed(3)}
                </span>
                {hoveredToken.category && (
                  <span className="ml-2 text-slate-400 text-[11px] capitalize">
                    ({hoveredToken.category} cue)
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Interactive Token Highlight Stream */}
          <div className="flex flex-wrap gap-2 p-4 rounded-xl bg-slate-950/70 border border-slate-800/80 leading-loose">
            {occlusion.attributions.map((a, idx) => {
              const k = Math.min(1, Math.abs(a.delta) / maxAbsDelta);
              const isPositiveDelta = a.delta >= 0;
              const bgColor = isPositiveDelta
                ? `rgba(244, 63, 94, ${0.12 + 0.5 * k})`
                : `rgba(56, 189, 248, ${0.12 + 0.5 * k})`;
              const borderColor = isPositiveDelta
                ? `rgba(244, 63, 94, ${0.3 + 0.4 * k})`
                : `rgba(56, 189, 248, ${0.3 + 0.4 * k})`;
              const textColor = isPositiveDelta ? "#fecdd3" : "#bae6fd";

              return (
                <span
                  key={idx}
                  onMouseEnter={() =>
                    setHoveredToken({
                      token: a.token,
                      delta: a.delta,
                      category: a.category,
                    })
                  }
                  onMouseLeave={() => setHoveredToken(null)}
                  style={{
                    backgroundColor: bgColor,
                    borderColor: borderColor,
                    color: textColor,
                  }}
                  className="px-2.5 py-1 rounded-md text-sm border font-medium cursor-pointer transition-all duration-200 hover:scale-105 select-none shadow-sm"
                  title={`Token: ${a.token} | Delta: ${a.delta > 0 ? "+" : ""}${a.delta.toFixed(3)}`}
                >
                  {a.token}
                </span>
              );
            })}
          </div>

          {/* Top Drivers breakdown */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
            <div className="p-3.5 rounded-xl bg-slate-950/50 border border-rose-950/50 space-y-2">
              <span className="text-xs font-semibold text-rose-300 uppercase tracking-wider flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-rose-500 inline-block" />
                Top Sarcasm Influencers (Δ &gt; 0)
              </span>
              <div className="flex flex-wrap gap-1.5">
                {occlusion.top_sarcastic.length > 0 ? (
                  occlusion.top_sarcastic.map((tok, i) => (
                    <span
                      key={i}
                      className="px-2 py-0.5 rounded bg-rose-950/60 border border-rose-800/40 text-rose-200 text-xs font-mono"
                    >
                      {tok.token} (+{tok.delta.toFixed(3)})
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-slate-500">None detected</span>
                )}
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-950/50 border border-sky-950/50 space-y-2">
              <span className="text-xs font-semibold text-sky-300 uppercase tracking-wider flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-sky-500 inline-block" />
                Top Literal Anchors (Δ &lt; 0)
              </span>
              <div className="flex flex-wrap gap-1.5">
                {occlusion.top_literal.length > 0 ? (
                  occlusion.top_literal.map((tok, i) => (
                    <span
                      key={i}
                      className="px-2 py-0.5 rounded bg-sky-950/60 border border-sky-800/40 text-sky-200 text-xs font-mono"
                    >
                      {tok.token} ({tok.delta.toFixed(3)})
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-slate-500">None detected</span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
