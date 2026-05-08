'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { Upload, FileText, Activity, Clock, CheckCircle, AlertCircle } from 'lucide-react';
import TherapeuticBlueprintTable from '../../components/TherapeuticBlueprintTable';

export default function PatientPortal() {
  const [step, setStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [blueprint, setBlueprint] = useState<any>(null);
  const [agreed, setAgreed] = useState(false);

  // Form State
  const [formData, setFormData] = useState({
    targetDisease: '',
    prescriptions: '',
    sleepSchedule: '',
  });
  const [healthReport, setHealthReport] = useState<File | null>(null);
  const [genomicVcf, setGenomicVcf] = useState<File | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, setter: (f: File) => void) => {
    if (e.target.files && e.target.files[0]) {
      setter(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!agreed) {
      setError("You must agree to the terms and conditions.");
      return;
    }
    if (!healthReport) {
      setError("Please upload a health report.");
      return;
    }

    setIsLoading(true);
    setError(null);

    const data = new FormData();
    data.append('target_disease', formData.targetDisease);
    data.append('current_prescriptions', formData.prescriptions);
    data.append('sleep_schedule', formData.sleepSchedule);
    data.append('health_report', healthReport);
    if (genomicVcf) data.append('genomic_vcf', genomicVcf);

    try {
      const response = await axios.post('/api/v1/generate_blueprint', data);
      setBlueprint(response.data);
      setStep(3);
    } catch (err) {
      setError("An error occurred while initializing the ZANE engine. Please try again.");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans p-8">
      <div className="max-w-4xl mx-auto">
        <header className="mb-12">
          <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
            ZANE N=1 Patient Portal
          </h1>
          <p className="text-gray-400 mt-2">Autonomous Molecular Engineering for Personalized Longevity</p>
        </header>

        <AnimatePresence mode="wait">
          {step === 1 && (
            <motion.div
              key="step1"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-8"
            >
              <div className="bg-zinc-900 border border-zinc-800 p-8 rounded-2xl shadow-xl">
                <h2 className="text-2xl font-semibold mb-6 flex items-center">
                  <FileText className="mr-2 text-blue-400" /> Terms and Conditions
                </h2>
                <div className="h-48 overflow-y-auto bg-black p-4 rounded border border-zinc-800 text-sm text-gray-400 mb-6">
                  <p className="mb-4 font-bold text-red-400">DISCLAIMER: EXPERIMENTAL PROTOCOL</p>
                  <p className="mb-2">1. ZANE is an experimental computational platform. The molecules and therapeutic blueprints generated are for research purposes only.</p>
                  <p className="mb-2">2. By using this portal, you acknowledge that ZANE's outputs have not been evaluated by the FDA for clinical use in humans.</p>
                  <p className="mb-2">3. Cosmic Hydra and ZANE developers assume no responsibility for health outcomes resulting from the use of this data.</p>
                  <p className="mb-2">4. You represent that you have the legal right to provide the uploaded health and genomic data.</p>
                </div>
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input 
                    type="checkbox" 
                    checked={agreed} 
                    onChange={(e) => setAgreed(e.target.checked)}
                    className="form-checkbox h-5 w-5 text-blue-500 rounded bg-zinc-800 border-zinc-700"
                  />
                  <span>I agree to the Terms and Conditions and understand the experimental nature of ZANE.</span>
                </label>
                <button
                  disabled={!agreed}
                  onClick={() => setStep(2)}
                  className="mt-8 w-full py-4 bg-gradient-to-r from-blue-600 to-emerald-600 rounded-xl font-bold hover:opacity-90 disabled:opacity-50 transition-all"
                >
                  Proceed to Intake
                </button>
              </div>
            </motion.div>
          )}

          {step === 2 && (
            <motion.form
              key="step2"
              onSubmit={handleSubmit}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-8"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Clinical Inputs */}
                <div className="bg-zinc-900 border border-zinc-800 p-8 rounded-2xl shadow-xl space-y-6">
                  <h2 className="text-2xl font-semibold mb-6 flex items-center">
                    <Activity className="mr-2 text-emerald-400" /> Clinical Data
                  </h2>
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-1">Target Disease / Purpose</label>
                    <input
                      type="text"
                      name="targetDisease"
                      value={formData.targetDisease}
                      onChange={handleInputChange}
                      placeholder="e.g. Type 2 Diabetes"
                      className="w-full bg-black border border-zinc-700 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 outline-none"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-1">Current Prescriptions</label>
                    <textarea
                      name="prescriptions"
                      value={formData.prescriptions}
                      onChange={handleInputChange}
                      placeholder="List all active medications..."
                      className="w-full bg-black border border-zinc-700 rounded-lg p-3 h-24 focus:ring-2 focus:ring-blue-500 outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-1">Lifestyle / Sleep Schedule</label>
                    <input
                      type="text"
                      name="sleepSchedule"
                      value={formData.sleepSchedule}
                      onChange={handleInputChange}
                      placeholder="e.g. 11 PM - 7 AM"
                      className="w-full bg-black border border-zinc-700 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 outline-none"
                    />
                  </div>
                </div>

                {/* Uploads */}
                <div className="bg-zinc-900 border border-zinc-800 p-8 rounded-2xl shadow-xl space-y-6">
                  <h2 className="text-2xl font-semibold mb-6 flex items-center">
                    <Upload className="mr-2 text-blue-400" /> Multi-Omic Uploads
                  </h2>
                  <div className="border-2 border-dashed border-zinc-700 rounded-xl p-6 text-center hover:border-blue-500 transition-colors">
                    <input
                      type="file"
                      id="healthReport"
                      className="hidden"
                      onChange={(e) => handleFileChange(e, setHealthReport)}
                    />
                    <label htmlFor="healthReport" className="cursor-pointer">
                      <FileText className="mx-auto h-12 w-12 text-zinc-500 mb-4" />
                      <p className="font-medium">{healthReport ? healthReport.name : 'Upload Health Report (PDF)'}</p>
                      <p className="text-xs text-gray-500 mt-2">Drag and drop or click to browse</p>
                    </label>
                  </div>

                  <div className="border-2 border-dashed border-zinc-700 rounded-xl p-6 text-center hover:border-emerald-500 transition-colors">
                    <input
                      type="file"
                      id="genomicVcf"
                      className="hidden"
                      onChange={(e) => handleFileChange(e, setGenomicVcf)}
                    />
                    <label htmlFor="genomicVcf" className="cursor-pointer">
                      <Activity className="mx-auto h-12 w-12 text-zinc-500 mb-4" />
                      <p className="font-medium">{genomicVcf ? genomicVcf.name : 'Upload Genomic VCF'}</p>
                      <p className="text-xs text-gray-500 mt-2">Optional genome sequencing data</p>
                    </label>
                  </div>
                </div>
              </div>

              {error && (
                <div className="bg-red-900/30 border border-red-500 text-red-200 p-4 rounded-lg flex items-center">
                  <AlertCircle className="mr-2" /> {error}
                </div>
              )}

              <button
                type="submit"
                disabled={isLoading}
                className="w-full py-4 bg-blue-600 rounded-xl font-bold hover:bg-blue-700 disabled:opacity-50 transition-all flex justify-center items-center"
              >
                {isLoading ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                      className="mr-3"
                    >
                      <Clock className="h-5 w-5" />
                    </motion.div>
                    Initializing ZANE Zero-Mortality Engine...
                  </>
                ) : (
                  'Generate N=1 Blueprint'
                )}
              </button>
            </motion.form>
          )}

          {step === 3 && blueprint && (
            <motion.div
              key="step3"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-8"
            >
              <div className="bg-zinc-900 border border-zinc-800 p-8 rounded-2xl shadow-xl">
                <div className="flex items-center justify-between mb-8">
                  <h2 className="text-2xl font-semibold flex items-center">
                    <CheckCircle className="mr-2 text-emerald-400" /> Therapeutic Blueprint Generated
                  </h2>
                  <button 
                    onClick={() => setStep(2)}
                    className="text-sm text-blue-400 hover:underline"
                  >
                    Run New Analysis
                  </button>
                </div>
                
                <TherapeuticBlueprintTable data={blueprint} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
