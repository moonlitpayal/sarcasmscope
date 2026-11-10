import { useState } from "react";
import { QuantumOrb } from "./components/QuantumOrb";
import { StudioView } from "./components/StudioView";
import { RobustnessView } from "./components/RobustnessView";
import { BatchView } from "./components/BatchView";
import { ProtocolView } from "./components/ProtocolView";
import { Sparkles, ShieldCheck, Layers, Settings2, Terminal } from "lucide-react";
import { TRAINED_TEMPERATURE, DEFAULT_ABSTAIN_THRESHOLD } from "./engine/hybridFusion";

export function App() {
  const [activeTab, setActiveTab] = useState<"studio" | "robustness" | "batch" | "protocol">("studio");
  const [temperature, setTemperature] = useState<number>(TRAINED_TEMPERATURE);
  const [abstainThreshold, setAbstainThreshold] = useState<number>(DEFAULT_ABSTAIN_THRESHOLD);

  // Cross-tab state sharing
  const [robustnessSample, setRobustnessSample] = useState<{ text: string; context: string }>({
    text: "वाह क्या बात है, 4 घंटे ट्रेन लेट करने के लिए रेलवे को 21 तोपों की सलामी! 🙄",
    context: "ट्रेन 4 घंटे लेट थी",
  });

  const handleSendToRobustness = (text: string, context: string) => {
    setRobustnessSample({ text, context });
    setActiveTab("robustness");
  };

  return (
    <div className="min-h-screen bg-[#07080c] text-slate-200 flex flex-col selection:bg-indigo-500/30 selection:text-indigo-200">
      {/* 3D Ambient Quantum Orb Visualizer Header */}
      <div className="relative border-b border-white/5 bg-gradient-to-b from-[#111322] via-[#07080c] to-[#07080c]">
        <QuantumOrb />

        <div className="max-w-6xl mx-auto px-4 sm:px-6 pb-6 text-center -mt-4 relative z-10 space-y-2">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-xs font-medium tracking-wide">
            <Sparkles className="w-3.5 h-3.5" />
            <span>SARcasmScope Studio • Cognitive Sarcasm Intelligence</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight bg-gradient-to-b from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
            Multilingual Sarcasm & Incongruity Engine
          </h1>

          <p className="text-xs sm:text-sm text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Calibrated, explainable sarcasm detection for code-mixed Indian-language text (English • Hindi • Hinglish • Marathi • Telugu). It selectively says <em>"uncertain"</em> instead of guessing when confidence is low.
          </p>

          {/* Navigation Tabs */}
          <div className="pt-4 flex justify-center">
            <div className="inline-flex p-1 rounded-xl bg-slate-900/90 border border-white/10 shadow-lg">
              <button
                onClick={() => setActiveTab("studio")}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  activeTab === "studio"
                    ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/40"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                }`}
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Studio Analyzer</span>
              </button>

              <button
                onClick={() => setActiveTab("robustness")}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  activeTab === "robustness"
                    ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/40"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                }`}
              >
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>Robustness Suite</span>
              </button>

              <button
                onClick={() => setActiveTab("batch")}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  activeTab === "batch"
                    ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/40"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Batch Evaluation</span>
              </button>

              <button
                onClick={() => setActiveTab("protocol")}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  activeTab === "protocol"
                    ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/40"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                }`}
              >
                <Settings2 className="w-3.5 h-3.5" />
                <span>Calibration & Protocol</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-6 flex-1 w-full">
        {activeTab === "studio" && (
          <StudioView
            temperature={temperature}
            abstainThreshold={abstainThreshold}
            onSelectSampleForRobustness={handleSendToRobustness}
          />
        )}

        {activeTab === "robustness" && (
          <RobustnessView
            initialText={robustnessSample.text}
            initialContext={robustnessSample.context}
            temperature={temperature}
            onSendToStudio={(text) => {
              setRobustnessSample((prev) => ({ ...prev, text }));
              setActiveTab("studio");
            }}
          />
        )}

        {activeTab === "batch" && (
          <BatchView temperature={temperature} abstainThreshold={abstainThreshold} />
        )}

        {activeTab === "protocol" && (
          <ProtocolView
            temperature={temperature}
            setTemperature={setTemperature}
            abstainThreshold={abstainThreshold}
            setAbstainThreshold={setAbstainThreshold}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 py-4 px-6 text-center text-xs text-slate-500">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>
            SARcasmScope • Cognitive Sarcasm Detection for Code-Mixed Indian Text
          </span>
          <div className="flex items-center gap-4 text-slate-400">
            <span className="flex items-center gap-1 font-mono">
              <Terminal className="w-3 h-3 text-indigo-400" />
              v1.0.0 React Rewrite
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
