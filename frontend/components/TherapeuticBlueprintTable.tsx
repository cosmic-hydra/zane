import React from 'react';
import { Pill, Clock, Zap, ShoppingBag, ArrowRight, ShieldCheck, Info } from 'lucide-react';

interface Compound {
  smiles: string;
  dosage: string;
  timing: string;
  purpose: string;
  toxicity_level: string;
}

interface BlueprintData {
  compounds: Compound[];
  commercial_match: {
    closest_drug: string;
    similarity: number;
    commercial_dose: string;
    extra_compounds: string[];
    missing_compounds: string[];
  };
}

export default function TherapeuticBlueprintTable({ data }: { data: BlueprintData }) {
  return (
    <div className="space-y-8 print:text-black">
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-zinc-800 text-gray-400 text-sm print:text-gray-600">
              <th className="pb-4 font-medium uppercase tracking-wider px-2">Compound</th>
              <th className="pb-4 font-medium uppercase tracking-wider px-2">Dosage</th>
              <th className="pb-4 font-medium uppercase tracking-wider px-2">Timing</th>
              <th className="pb-4 font-medium uppercase tracking-wider px-2">Purpose / Toxicity</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800 print:divide-gray-200">
            {data.compounds.map((c, idx) => (
              <tr key={idx} className="group hover:bg-zinc-800/50 print:hover:bg-transparent">
                <td className="py-4 px-2 max-w-[200px]">
                  <div className="bg-black p-2 rounded border border-zinc-800 font-mono text-[10px] text-blue-300 break-all print:bg-gray-100 print:text-blue-900 print:border-gray-300">
                    {c.smiles}
                  </div>
                </td>
                <td className="py-4 px-2">
                  <div className="flex items-center font-semibold text-emerald-400 print:text-emerald-700">
                    <Pill className="mr-2 h-4 w-4 shrink-0" /> {c.dosage}
                  </div>
                </td>
                <td className="py-4 px-2">
                  <div className="flex items-center text-zinc-300 print:text-gray-800">
                    <Clock className="mr-2 h-4 w-4 text-blue-400 shrink-0 print:text-blue-700" /> {c.timing}
                  </div>
                </td>
                <td className="py-4 px-2">
                  <div className="space-y-1">
                    <div className="flex items-start text-xs text-zinc-400 print:text-gray-600">
                      <Zap className="mr-2 h-4 w-4 text-yellow-500 shrink-0" />
                      {c.purpose}
                    </div>
                    <div className="flex items-center text-[10px] font-bold text-emerald-500 uppercase tracking-tighter">
                      <ShieldCheck className="mr-1 h-3 w-3" /> Toxicity: {c.toxicity_level}
                    </div>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {/* Industry-Level Equivalent Comparison */}
      <div className="bg-blue-900/10 border border-blue-900/30 rounded-2xl p-6 print:border-gray-400 print:bg-white">
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 gap-4">
          <div className="flex items-center">
            <div className="bg-blue-600/20 p-3 rounded-xl mr-4 print:bg-blue-100">
              <ShoppingBag className="h-8 w-8 text-blue-400 print:text-blue-600" />
            </div>
            <div>
              <h4 className="text-xs font-bold text-blue-400 uppercase tracking-widest print:text-blue-700">Industry-Level Equivalent Drug</h4>
              <div className="flex items-center mt-1">
                <span className="text-2xl font-bold text-white mr-3 print:text-black">{data.commercial_match.closest_drug}</span>
                <span className="bg-blue-900 text-blue-200 text-xs px-2 py-0.5 rounded-full font-bold border border-blue-700 print:bg-blue-100 print:text-blue-800 print:border-blue-300">
                  {(data.commercial_match.similarity * 100).toFixed(1)}% Core Match
                </span>
              </div>
            </div>
          </div>
          <div className="text-right">
            <p className="text-gray-500 uppercase text-[10px] font-bold">Standard Commercial Dose</p>
            <p className="text-zinc-300 font-medium print:text-gray-800">{data.commercial_match.commercial_dose}</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-6 border-t border-blue-900/20 print:border-gray-200">
          <div>
            <h5 className="text-sm font-bold text-emerald-400 mb-3 flex items-center print:text-emerald-700">
              <Info className="mr-2 h-4 w-4" /> Extra Compounds in ZANE Design
            </h5>
            <ul className="space-y-2">
              {data.commercial_match.extra_compounds.map((ec, i) => (
                <li key={i} className="text-[10px] font-mono bg-emerald-900/10 border border-emerald-900/30 p-2 rounded text-emerald-300 break-all print:bg-emerald-50 print:text-emerald-800 print:border-emerald-200">
                  {ec}
                </li>
              ))}
            </ul>
            <p className="text-[10px] text-zinc-500 mt-2 italic">These are added for N=1 metabolic synergy and toxicity mitigation.</p>
          </div>
          <div>
            <h5 className="text-sm font-bold text-red-400 mb-3 flex items-center print:text-red-700">
              <Info className="mr-2 h-4 w-4" /> Missing Compounds (vs. Commercial)
            </h5>
            <ul className="space-y-2">
              {data.commercial_match.missing_compounds.map((mc, i) => (
                <li key={i} className="text-xs bg-red-900/10 border border-red-900/30 p-2 rounded text-red-300 print:bg-red-50 print:text-red-800 print:border-red-200">
                  {mc}
                </li>
              ))}
            </ul>
            <p className="text-[10px] text-zinc-500 mt-2 italic">Non-functional excipients removed to achieve higher purity and lower toxicity.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
