import React from 'react';
import { Pill, Clock, Zap, ShoppingBag, ArrowRight } from 'lucide-react';

interface BlueprintData {
  compound_smiles: string;
  dosage: string;
  administration_time: string;
  mechanism_of_action: string;
  commercial_match: {
    closest_drug: string;
    similarity: number;
    commercial_dose: string;
  };
}

export default function TherapeuticBlueprintTable({ data }: { data: BlueprintData }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="border-b border-zinc-800 text-gray-400 text-sm">
            <th className="pb-4 font-medium uppercase tracking-wider">Compound Structure</th>
            <th className="pb-4 font-medium uppercase tracking-wider">Precise N=1 Dosage</th>
            <th className="pb-4 font-medium uppercase tracking-wider">Chronobiological Time</th>
            <th className="pb-4 font-medium uppercase tracking-wider">Mechanism of Action</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800">
          <tr className="group">
            <td className="py-6 pr-4">
              <div className="bg-black p-3 rounded border border-zinc-800 font-mono text-xs text-blue-300 break-all">
                {data.compound_smiles}
              </div>
            </td>
            <td className="py-6 pr-4">
              <div className="flex items-center font-semibold text-lg text-emerald-400">
                <Pill className="mr-2 h-5 w-5" /> {data.dosage}
              </div>
            </td>
            <td className="py-6 pr-4">
              <div className="flex items-center text-zinc-300">
                <Clock className="mr-2 h-5 w-5 text-blue-400" /> {data.administration_time}
              </div>
            </td>
            <td className="py-6">
              <div className="flex items-start text-sm text-zinc-400">
                <Zap className="mr-2 h-5 w-5 text-yellow-500 shrink-0" />
                {data.mechanism_of_action}
              </div>
            </td>
          </tr>
          
          {/* Commercial Equivalent Section */}
          <tr className="bg-blue-900/10">
            <td colSpan={4} className="py-4 px-6 rounded-b-xl">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="bg-blue-600/20 p-2 rounded-lg mr-4">
                    <ShoppingBag className="h-6 w-6 text-blue-400" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-blue-400 uppercase tracking-widest">Closest Commercial Equivalent</h4>
                    <div className="flex items-center mt-1">
                      <span className="text-xl font-bold text-white mr-3">{data.commercial_match.closest_drug}</span>
                      <span className="bg-blue-900 text-blue-200 text-xs px-2 py-0.5 rounded-full font-bold border border-blue-700">
                        {(data.commercial_match.similarity * 100).toFixed(1)}% Match
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-8 text-sm">
                  <div className="text-right">
                    <p className="text-gray-500 uppercase text-[10px] font-bold">Standard Commercial Dose</p>
                    <p className="text-zinc-300 font-medium">{data.commercial_match.commercial_dose}</p>
                  </div>
                  <div className="h-8 w-px bg-zinc-800"></div>
                  <button className="flex items-center text-blue-400 hover:text-blue-300 font-bold transition-colors">
                    View Pharmanet Listing <ArrowRight className="ml-1 h-4 w-4" />
                  </button>
                </div>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
