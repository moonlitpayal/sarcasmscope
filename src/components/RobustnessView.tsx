import { useState } from "react";
import { RobustnessResult, PRESET_SAMPLES } from "../engine/sarcasmEngine";
import { perturbHybridAll } from "../engine/hybridFusion";
import { ShieldCheck, AlertOctagon, RefreshCw, Check, ArrowRight } from "lucide-react";

interface RobustnessViewProps {
  initialText?: string;
  initialContext?: string;
  temperature: number;
  onSendToStudio?: (text: string) => void;
}

export function RobustnessView({
  initialText = "वाह क्या बात है, 4 घंटे ट्रेन लेट करने के लिए रेलवे को 21 तोपों की सलामी! 🙄",
  initialContext = "ट्रेन 4 घंटे लेट थी",
  temperature,
  onSendToStudio,
}: RobustnessViewProps) {
  const [text, setText] = useState(initialText);
  const [context, setContext] = useState(initialContext);
  const [data, setData] = useState<RobustnessResult>(() =>
    perturbHybridAll(initialText, initialContext, temperature)
  );

  const handleRun = () => {
    if (!text.trim()) return;
    setData(perturbHybridAll(text, context, temperature));
  };

  const flippedCount = data.results.filter((r) => r.flipped).length;

  return (
    <div className="space-y-6">
      {/* Overview Header Card */}
      <div className="bg-slate-900/70 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-base font-semibold text-slate-100">
              <ShieldCheck className="w-5 h-5 text-indigo-400" />
              Controlled Perturbation & Noise Robustness Check
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-2xl">
              Real-world Indian-language text on social media features erratic spelling, repeated
              letters, code-mixing, missing emoji, and noise. This suite evaluates whether the model
              verdict survives noise or flips dangerously.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="px-4 py-2 rounded-xl bg-slate-950 border border-slate-800 text-center">
              <div className="text-[11px] text-slate-400 uppercase font-medium">Stability Score</div>
              <div className="text-xl font-bold font-mono text-emerald-400">
                {data.stability_score}%
              </div>
            </div>

            <div className="px-4 py-2 rounded-xl bg-slate-950 border border-slate-800 text-center">
              <div className="text-[11px] text-slate-400 uppercase font-medium">Decision Flips</div>
              <div
                className={`text-xl font-bold font-mono ${
                  flippedCount > 0 ? "text-rose-400" : "text-emerald-400"
                }`}
              >
                {flippedCount} / {data.results.length}
              </div>
            </div>
          </div>
        </div>

        {/* Input area */}
        <div className="pt-2 border-t border-white/5 grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="md:col-span-2 space-y-1">
            <label className="text-xs font-semibold text-slate-300">Target Statement Under Test</label>
            <input
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Context (Optional)</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={context}
                onChange={(e) => setContext(e.target.value)}
                placeholder="Preceding discourse"
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-indigo-500"
              />
              <button
                onClick={handleRun}
                className="px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center gap-1.5 cursor-pointer whitespace-nowrap shadow-md shadow-indigo-600/30"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Perturb
              </button>
            </div>
          </div>
        </div>

        {/* Quick sample loader */}
        <div className="flex flex-wrap items-center gap-2 pt-1 text-xs">
          <span className="text-slate-500 font-medium">Test other samples:</span>
          {PRESET_SAMPLES.slice(0, 5).map((p) => (
            <button
              key={p.name}
              onClick={() => {
                setText(p.statement);
                setContext(p.context);
                setData(perturbHybridAll(p.statement, p.context, temperature));
              }}
              className="px-2.5 py-1 rounded-md bg-slate-800/60 hover:bg-slate-700/80 border border-slate-800 text-slate-300 cursor-pointer"
            >
              {p.name}
            </button>
          ))}
        </div>
      </div>

      {/* Perturbations Results Table */}
      <div className="bg-slate-900/70 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden shadow-xl">
        <div className="p-4 border-b border-white/5 flex items-center justify-between">
          <div className="text-sm font-semibold text-slate-200">
            Perturbation Suite Results
          </div>
          <div className="text-xs text-slate-400 font-mono">
            Baseline: P = <strong>{data.baseline_p_sarcastic.toFixed(3)}</strong> ({data.baseline_label})
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm border-collapse">
            <thead>
              <tr className="border-b border-white/5 bg-slate-950/60 text-slate-400 text-xs font-semibold uppercase tracking-wider">
                <th className="py-3 px-4">Perturbation</th>
                <th className="py-3 px-4">Description</th>
                <th className="py-3 px-4">Text After Change</th>
                <th className="py-3 px-4 text-center">P(Sarcastic)</th>
                <th className="py-3 px-4 text-center">Δ Delta</th>
                <th className="py-3 px-4 text-right">Verdict Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {data.results.map((row) => {
                return (
                  <tr
                    key={row.perturbation}
                    className={`hover:bg-slate-800/30 transition-colors ${
                      row.flipped ? "bg-rose-950/20" : ""
                    }`}
                  >
                    <td className="py-3.5 px-4 font-mono text-xs text-indigo-300 font-semibold whitespace-nowrap">
                      {row.perturbation}
                    </td>

                    <td className="py-3.5 px-4 text-xs text-slate-400">
                      {row.description}
                    </td>

                    <td className="py-3.5 px-4 text-xs text-slate-200 max-w-md break-words font-mono">
                      <div className="flex items-center justify-between gap-2">
                        <span>{row.perturbed_text}</span>
                        {onSendToStudio && (
                          <button
                            onClick={() => onSendToStudio(row.perturbed_text)}
                            title="Analyze this perturbed version in Studio"
                            className="text-slate-500 hover:text-indigo-300 p-1 rounded hover:bg-slate-800 transition-colors cursor-pointer"
                          >
                            <ArrowRight className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    </td>

                    <td className="py-3.5 px-4 text-center font-mono font-semibold text-xs">
                      <span
                        className={
                          row.p_sarcastic >= 0.5 ? "text-rose-400" : "text-emerald-400"
                        }
                      >
                        {row.p_sarcastic.toFixed(3)}
                      </span>
                    </td>

                    <td className="py-3.5 px-4 text-center font-mono text-xs text-slate-400">
                      {row.delta >= 0 ? `+${row.delta.toFixed(3)}` : row.delta.toFixed(3)}
                    </td>

                    <td className="py-3.5 px-4 text-right">
                      {row.flipped ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-rose-500/20 border border-rose-500/40 text-rose-300 text-xs font-semibold animate-pulse">
                          <AlertOctagon className="w-3.5 h-3.5" />
                          Answer Flips!
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs font-medium">
                          <Check className="w-3.5 h-3.5" />
                          Preserved
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
