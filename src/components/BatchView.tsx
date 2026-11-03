import { useState } from "react";
import { predict, PredictionResult } from "../engine/sarcasmEngine";
import { Layers, Download, CheckCircle2, AlertTriangle, HelpCircle, Filter } from "lucide-react";

interface BatchViewProps {
  temperature: number;
  abstainThreshold: number;
}

const DEFAULT_BATCH_INPUT = `वाह, फिर से बारिश! 🙄
The train arrived exactly on time at Dadar station today.
Bohot badiya bhai, exam se ek raat pehle phone pe reels dekh, pakka top karega! 🙃
The technical documentation is clean, concise, and easy to follow.
Lai bhari bhau, 2 taas traffic madhe thevlya badal khup khup aabhar!
Our team solved the production issue within 10 minutes, outstanding effort.
Oh brilliant, another server outage on Friday evening.
आज की बैठक बहुत ही उत्पादक रही और समय पर समाप्त हो गई।
great, another rainy Monday with traffic jams everywhere...
खूप छान नियोजन आहे, अर्धा तास आधी येऊन पण लाईन मध्येच उभे आहोत!`;

export function BatchView({ temperature, abstainThreshold }: BatchViewProps) {
  const [textInput, setTextInput] = useState(DEFAULT_BATCH_INPUT);
  const [results, setResults] = useState<PredictionResult[]>(() => {
    return DEFAULT_BATCH_INPUT.split("\n")
      .map((s) => s.trim())
      .filter(Boolean)
      .map((t) => predict(t, "", temperature, abstainThreshold));
  });
  const [filterType, setFilterType] = useState<"all" | "sarcastic" | "literal" | "uncertain">("all");
  const [searchQuery, setSearchQuery] = useState("");

  const handleRunBatch = () => {
    const lines = textInput
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    if (!lines.length) return;

    const evaluated = lines.map((line) => predict(line, "", temperature, abstainThreshold));
    setResults(evaluated);
  };

  const handleExportCSV = () => {
    if (!results.length) return;
    const headers = [
      "Text",
      "Verdict",
      "P_Sarcastic",
      "Confidence",
      "Abstain",
      "Script",
      "Language",
      "CMI",
      "Incongruity_Index",
    ];
    const rows = results.map((r) => [
      `"${r.text.replace(/"/g, '""')}"`,
      r.verdict,
      r.p_sarcastic.toFixed(3),
      r.confidence.toFixed(3),
      r.abstain ? "YES" : "NO",
      `"${r.language.primary_script}"`,
      r.language.language,
      r.language.cmi,
      r.incongruity.incongruity_index,
    ]);

    const csvContent = [headers.join(","), ...rows.map((row) => row.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `sarcasmscope_batch_eval_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredResults = results.filter((r) => {
    if (filterType === "sarcastic" && (r.abstain || r.verdict !== "Sarcastic")) return false;
    if (filterType === "literal" && (r.abstain || r.verdict !== "Literal")) return false;
    if (filterType === "uncertain" && !r.abstain) return false;
    if (searchQuery && !r.text.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const sarcasticCount = results.filter((r) => !r.abstain && r.verdict === "Sarcastic").length;
  const literalCount = results.filter((r) => !r.abstain && r.verdict === "Literal").length;
  const uncertainCount = results.filter((r) => r.abstain).length;
  const avgConfidence = results.length
    ? results.reduce((acc, r) => acc + r.confidence, 0) / results.length
    : 0;

  return (
    <div className="space-y-6">
      {/* Batch Input Area */}
      <div className="bg-slate-900/70 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              <Layers className="w-5 h-5 text-indigo-400" />
              Batch Corpus Evaluator
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Input one sentence per line (up to 200 items) to benchmark multilingual sarcasm detection across full datasets.
            </p>
          </div>
          <div className="text-xs text-slate-400 font-mono">
            {textInput.split("\n").filter((l) => l.trim()).length} lines loaded
          </div>
        </div>

        <textarea
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          rows={5}
          placeholder="Enter one statement per line..."
          className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-3.5 text-slate-100 text-sm font-mono placeholder:text-slate-600 focus:outline-none focus:border-indigo-500"
        />

        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
          <button
            onClick={handleRunBatch}
            className="py-2.5 px-6 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs shadow-md shadow-indigo-600/30 flex items-center gap-2 cursor-pointer transition-all"
          >
            <span>Run Batch Evaluation</span>
          </button>

          <button
            onClick={handleExportCSV}
            disabled={!results.length}
            className="py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
          >
            <Download className="w-3.5 h-3.5" />
            Export CSV
          </button>
        </div>
      </div>

      {/* Batch Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-slate-900/70 border border-white/10 rounded-xl p-4">
          <div className="text-xs text-slate-400 uppercase font-medium">Total Evaluated</div>
          <div className="text-2xl font-bold font-mono text-slate-100 mt-1">
            {results.length}
          </div>
        </div>

        <div className="bg-slate-900/70 border border-rose-950/40 rounded-xl p-4">
          <div className="text-xs text-rose-300 uppercase font-medium flex items-center gap-1">
            <AlertTriangle className="w-3.5 h-3.5 text-rose-400" /> Sarcastic
          </div>
          <div className="text-2xl font-bold font-mono text-rose-400 mt-1">
            {sarcasticCount}{" "}
            <span className="text-xs text-slate-400 font-normal">
              ({results.length ? Math.round((sarcasticCount / results.length) * 100) : 0}%)
            </span>
          </div>
        </div>

        <div className="bg-slate-900/70 border border-emerald-950/40 rounded-xl p-4">
          <div className="text-xs text-emerald-300 uppercase font-medium flex items-center gap-1">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Literal
          </div>
          <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">
            {literalCount}{" "}
            <span className="text-xs text-slate-400 font-normal">
              ({results.length ? Math.round((literalCount / results.length) * 100) : 0}%)
            </span>
          </div>
        </div>

        <div className="bg-slate-900/70 border border-amber-950/40 rounded-xl p-4">
          <div className="text-xs text-amber-300 uppercase font-medium flex items-center gap-1">
            <HelpCircle className="w-3.5 h-3.5 text-amber-400" /> Uncertain (Abstain)
          </div>
          <div className="text-2xl font-bold font-mono text-amber-400 mt-1">
            {uncertainCount}{" "}
            <span className="text-xs text-slate-400 font-normal">
              (Avg conf: {avgConfidence.toFixed(2)})
            </span>
          </div>
        </div>
      </div>

      {/* Filter and Table Container */}
      <div className="bg-slate-900/70 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden shadow-xl">
        <div className="p-4 border-b border-white/5 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-slate-400" />
            <div className="flex gap-1">
              {(["all", "sarcastic", "literal", "uncertain"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setFilterType(t)}
                  className={`text-xs px-3 py-1.5 rounded-lg capitalize transition-colors cursor-pointer ${
                    filterType === t
                      ? "bg-indigo-600 text-white font-medium"
                      : "bg-slate-800/60 text-slate-400 hover:text-white"
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          <input
            type="text"
            placeholder="Search in batch results..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full sm:w-64 bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
          />
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm border-collapse">
            <thead>
              <tr className="border-b border-white/5 bg-slate-950/60 text-slate-400 text-xs font-semibold uppercase tracking-wider">
                <th className="py-3 px-4">Statement</th>
                <th className="py-3 px-4">Verdict</th>
                <th className="py-3 px-4 text-center">P(Sarcastic)</th>
                <th className="py-3 px-4 text-center">Confident?</th>
                <th className="py-3 px-4">Script / CMI</th>
                <th className="py-3 px-4 text-right">Incongruity</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {filteredResults.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                  <td className="py-3.5 px-4 text-xs text-slate-200 max-w-md font-mono">
                    {row.text}
                  </td>

                  <td className="py-3.5 px-4">
                    {row.abstain ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-amber-500/20 border border-amber-500/30 text-amber-300 text-xs font-medium">
                        Uncertain
                      </span>
                    ) : row.verdict === "Sarcastic" ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-rose-500/20 border border-rose-500/30 text-rose-300 text-xs font-semibold">
                        Sarcastic
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 text-xs font-medium">
                        Literal
                      </span>
                    )}
                  </td>

                  <td className="py-3.5 px-4 text-center font-mono text-xs">
                    <span
                      className={
                        row.p_sarcastic >= 0.5 ? "text-rose-400 font-semibold" : "text-emerald-400"
                      }
                    >
                      {row.p_sarcastic.toFixed(3)}
                    </span>
                  </td>

                  <td className="py-3.5 px-4 text-center text-xs">
                    {row.abstain ? (
                      <span className="text-amber-400 font-semibold">No (Abstain)</span>
                    ) : (
                      <span className="text-emerald-400 font-mono">
                        Yes ({row.confidence.toFixed(2)})
                      </span>
                    )}
                  </td>

                  <td className="py-3.5 px-4 text-xs text-slate-400">
                    <span className="text-slate-300">{row.language.primary_script}</span>
                    {row.language.cmi > 0 && (
                      <span className="ml-1 text-[11px] text-indigo-400">
                        ({row.language.cmi}%)
                      </span>
                    )}
                  </td>

                  <td className="py-3.5 px-4 text-right font-mono text-xs text-indigo-300">
                    {row.incongruity.incongruity_index.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
