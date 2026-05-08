'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { Upload, FileText, Activity, Clock, CheckCircle, AlertCircle, User, MapPin, Printer, Mail, Phone } from 'lucide-react';
import TherapeuticBlueprintTable from '../../components/TherapeuticBlueprintTable';

export default function PatientPortal() {
  const [step, setStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [blueprint, setBlueprint] = useState<any>(null);
  const [agreed, setAgreed] = useState(false);

  // Form State
  const [formData, setFormData] = useState({
    name: '',
    dob: '',
    phone: '',
    email: '',
    location: '',
    targetPurpose: '',
    treatments: '',
    lifestyle: '',
    hereditary: '',
  });
  const [healthReport, setHealthReport] = useState<File | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setHealthReport(e.target.files[0]);
    }
  };

  const isFormValid = () => {
    return (
      formData.name.trim() !== '' &&
      formData.dob.trim() !== '' &&
      formData.phone.trim() !== '' &&
      formData.email.trim() !== '' &&
      formData.location.trim() !== '' &&
      formData.targetPurpose.trim() !== '' &&
      formData.lifestyle.trim() !== '' &&
      healthReport !== null
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!agreed) {
      setError("You must agree to the terms and conditions.");
      return;
    }
    if (!isFormValid()) {
      setError("Please fill in all mandatory fields and upload a health report.");
      return;
    }

    setIsLoading(true);
    setError(null);

    const data = new FormData();
    data.append('name', formData.name);
    data.append('dob', formData.dob);
    data.append('phone', formData.phone);
    data.append('email', formData.email);
    data.append('location', formData.location);
    data.append('target_purpose', formData.targetPurpose);
    data.append('current_treatments', formData.treatments);
    data.append('lifestyle', formData.lifestyle);
    data.append('hereditary_problems', formData.hereditary);
    data.append('health_report', healthReport as File);

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

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans p-8 print:bg-white print:p-0">
      <div className="max-w-4xl mx-auto">
        <header className="mb-12 flex justify-between items-start print:mb-6">
          <div>
            <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400 print:text-black print:from-black print:to-black">
              ZANE N=1 Patient Portal
            </h1>
            <p className="text-gray-400 mt-2 print:text-gray-600">Autonomous Molecular Engineering for Personalized Longevity</p>
          </div>
          {step === 3 && (
            <button
              onClick={handlePrint}
              className="bg-zinc-800 p-3 rounded-lg hover:bg-zinc-700 transition-colors print:hidden flex items-center"
            >
              <Printer className="mr-2 h-5 w-5" /> Print Report
            </button>
          )}
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
                  <p className="mb-2">4. Personal data provided will be used solely for the generation of the therapeutic blueprint. Email and phone number are not passed to the core molecular engine.</p>
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
                {/* Personal Information (Mandatory) */}
                <div className="bg-zinc-900 border border-zinc-800 p-8 rounded-2xl shadow-xl space-y-6">
                  <h2 className="text-2xl font-semibold mb-6 flex items-center">
                    <User className="mr-2 text-blue-400" /> Personal Information
                  </h2>
                  <div className="grid grid-cols-1 gap-4">
                    <input type="text" name="name" placeholder="Full Name *" value={formData.name} onChange={handleInputChange} className="input-field" required />
                    <input type="text" name="dob" placeholder="Date of Birth (YYYY-MM-DD) *" value={formData.dob} onChange={handleInputChange} className="input-field" required />
                    <div className="flex gap-4">
                      <input type="tel" name="phone" placeholder="Phone *" value={formData.phone} onChange={handleInputChange} className="input-field w-1/2" required />
                      <input type="email" name="email" placeholder="Email *" value={formData.email} onChange={handleInputChange} className="input-field w-1/2" required />
                    </div>
                    <input type="text" name="location" placeholder="Location (City, Country) *" value={formData.location} onChange={handleInputChange} className="input-field" required />
                  </div>
                </div>

                {/* Health Report Upload (Mandatory) */}
                <div className="bg-zinc-900 border border-zinc-800 p-8 rounded-2xl shadow-xl space-y-6">
                  <h2 className="text-2xl font-semibold mb-6 flex items-center">
                    <Upload className="mr-2 text-emerald-400" /> Health Data Upload
                  </h2>
                  <div className="border-2 border-dashed border-zinc-700 rounded-xl p-8 text-center hover:border-blue-500 transition-colors">
                    <input
                      type="file"
                      id="healthReport"
                      className="hidden"
                      onChange={handleFileChange}
                      accept=".pdf,.doc,.docx,image/*"
                    />
                    <label htmlFor="healthReport" className="cursor-pointer">
                      <FileText className="mx-auto h-12 w-12 text-zinc-500 mb-4" />
                      <p className="font-medium">{healthReport ? healthReport.name : 'Upload Health Report *'}</p>
                      <p className="text-xs text-gray-500 mt-2">PDF, Word, or Image files supported</p>
                    </label>
                  </div>
                </div>

                {/* Clinical Context */}
                <div className="bg-zinc-900 border border-zinc-800 p-8 rounded-2xl shadow-xl space-y-6 md:col-span-2">
                  <h2 className="text-2xl font-semibold mb-6 flex items-center">
                    <Activity className="mr-2 text-purple-400" /> Bio-Clinical Context
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <label className="block text-sm font-medium text-gray-400">Purpose of Generation *</label>
                      <input type="text" name="targetPurpose" placeholder="e.g. Chronic Inflammation, Performance" value={formData.targetPurpose} onChange={handleInputChange} className="input-field" required />
                      
                      <label className="block text-sm font-medium text-gray-400">Current Medications / Treatments</label>
                      <textarea name="treatments" placeholder="Specify if undergoing any treatment..." value={formData.treatments} onChange={handleInputChange} className="input-field h-24" />
                    </div>
                    <div className="space-y-4">
                      <label className="block text-sm font-medium text-gray-400">Detailed Lifestyle *</label>
                      <textarea name="lifestyle" placeholder="Sleep, Diet, Stress levels, Exercise..." value={formData.lifestyle} onChange={handleInputChange} className="input-field h-24" required />

                      <label className="block text-sm font-medium text-gray-400">Hereditary Problems</label>
                      <textarea name="hereditary" placeholder="Family history of disease..." value={formData.hereditary} onChange={handleInputChange} className="input-field h-24" />
                    </div>
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
                className="w-full py-4 bg-gradient-to-r from-blue-600 to-emerald-600 rounded-xl font-bold hover:opacity-90 disabled:opacity-50 transition-all flex justify-center items-center shadow-lg shadow-blue-900/20"
              >
                {isLoading ? (
                  <>
                    <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: "linear" }} className="mr-3">
                      <Clock className="h-5 w-5" />
                    </motion.div>
                    ZANE Engine Analyzing N=1 Profiles...
                  </>
                ) : (
                  'GENERATE AUTONOMOUS DRUG BLUEPRINT'
                )}
              </button>
            </motion.form>
          )}

          {step === 3 && blueprint && (
            <motion.div
              key="step3"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-8 print:space-y-4"
            >
              <div className="bg-zinc-900 border border-zinc-800 p-8 rounded-2xl shadow-xl print:bg-white print:border-none print:shadow-none print:p-0">
                <div className="flex items-center justify-between mb-8 print:mb-4">
                  <div>
                    <h2 className="text-2xl font-semibold flex items-center print:text-black">
                      <CheckCircle className="mr-2 text-emerald-400" /> N=1 Therapeutic Blueprint
                    </h2>
                    <div className="flex gap-6 mt-2 text-xs text-gray-500 print:text-gray-700">
                      <span><strong>Patient:</strong> {formData.name}</span>
                      <span><strong>DOB:</strong> {formData.dob}</span>
                      <span><strong>Location:</strong> {formData.location}</span>
                    </div>
                  </div>
                  <button 
                    onClick={() => setStep(2)}
                    className="text-sm text-blue-400 hover:underline print:hidden"
                  >
                    Generate New
                  </button>
                </div>
                
                <TherapeuticBlueprintTable data={blueprint} />

                <div className="mt-12 pt-8 border-t border-zinc-800 text-center text-xs text-gray-500 print:mt-6 print:pt-4 print:text-gray-700">
                  <p>Certified by ZANE Autonomous Molecular Engineering Engine - v4.2.0-apex</p>
                  <p className="mt-1">Cryptographic ID: {Math.random().toString(36).substring(7).toUpperCase()}</p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <style jsx global>{`
        .input-field {
          width: 100%;
          background-color: black;
          border: 1px solid #27272a;
          border-radius: 0.5rem;
          padding: 0.75rem;
          outline: none;
          transition: all 0.2s;
        }
        .input-field:focus {
          border-color: #3b82f6;
          box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
        }
        @media print {
          .print-hidden { display: none !important; }
          body { background: white !important; color: black !important; }
        }
      `}</style>
    </div>
  );
}
