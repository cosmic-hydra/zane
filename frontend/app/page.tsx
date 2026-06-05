'use client';

import React, { useState, useEffect } from 'react';
import { Activity, Beaker, Shield, Cpu, TrendingUp, Zap } from 'lucide-react';

interface PipelineStatus {
  status: string;
  version: string;
  uptime_seconds: number;
  active_jobs: number;
}

interface StatCard {
  label: string;
  value: string;
  trend: string;
  icon: React.ReactNode;
  color: string;
}

export default function Dashboard() {
  const [status, setStatus] = useState<PipelineStatus | null>(null);

  useEffect(() => {
    async function fetchStatus() {
      try {
        const res = await fetch('/api/health');
        if (res.ok) {
          setStatus(await res.json());
        }
      } catch {
        setStatus({
          status: 'demo',
          version: '2026.4.1',
          uptime_seconds: 0,
          active_jobs: 0,
        });
      }
    }
    fetchStatus();
  }, []);

  const stats: StatCard[] = [
    {
      label: 'Molecules Screened',
      value: '124,830',
      trend: '+12.4%',
      icon: <Beaker className="w-5 h-5" />,
      color: 'from-cyan-500 to-blue-500',
    },
    {
      label: 'Hit Rate',
      value: '18.3%',
      trend: '+2.1%',
      icon: <TrendingUp className="w-5 h-5" />,
      color: 'from-green-500 to-emerald-500',
    },
    {
      label: 'Safety Gates Passed',
      value: '97.2%',
      trend: '+0.8%',
      icon: <Shield className="w-5 h-5" />,
      color: 'from-violet-500 to-purple-500',
    },
    {
      label: 'Active Pipelines',
      value: '8',
      trend: 'stable',
      icon: <Cpu className="w-5 h-5" />,
      color: 'from-orange-500 to-red-500',
    },
    {
      label: 'Avg QED Score',
      value: '0.742',
      trend: '+0.008',
      icon: <Activity className="w-5 h-5" />,
      color: 'from-pink-500 to-rose-500',
    },
    {
      label: 'Inference Latency',
      value: '61.5ms',
      trend: '-4.2ms',
      icon: <Zap className="w-5 h-5" />,
      color: 'from-yellow-500 to-amber-500',
    },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Discovery Dashboard</h1>
        <p className="text-gray-400 mt-1">
          Real-time overview of the ZANE drug discovery pipeline
          {status && (
            <span className="ml-3 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-900 text-green-300">
              {status.status === 'demo' ? 'Demo Mode' : 'Connected'}
            </span>
          )}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {stats.map((stat) => (
          <div
            key={stat.label}
            className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 hover:border-gray-600 transition-colors"
          >
            <div className="flex items-center justify-between mb-4">
              <div className={`p-2 rounded-lg bg-gradient-to-r ${stat.color} bg-opacity-20`}>
                {stat.icon}
              </div>
              <span className="text-sm text-green-400">{stat.trend}</span>
            </div>
            <p className="text-2xl font-bold">{stat.value}</p>
            <p className="text-sm text-gray-400 mt-1">{stat.label}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-4">Pipeline Stages</h2>
          <div className="space-y-3">
            {[
              { name: 'Data Acquisition', status: 'Running', progress: 92 },
              { name: 'Molecular Featurization', status: 'Running', progress: 88 },
              { name: 'GNN Training', status: 'Running', progress: 76 },
              { name: 'ADMET Screening', status: 'Running', progress: 95 },
              { name: 'Docking Simulation', status: 'Queued', progress: 0 },
            ].map((stage) => (
              <div key={stage.name} className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex justify-between mb-1">
                    <span className="text-sm">{stage.name}</span>
                    <span className="text-xs text-gray-400">{stage.status}</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-cyan-500 to-blue-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${stage.progress}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-4">Recent Activity</h2>
          <div className="space-y-4">
            {[
              { time: '2m ago', event: 'GNN model checkpoint saved (epoch 98)' },
              { time: '5m ago', event: 'ADMET gate: 42 candidates promoted' },
              { time: '12m ago', event: 'New data batch ingested (1,240 molecules)' },
              { time: '18m ago', event: 'Toxicity gate: 3 candidates flagged for review' },
              { time: '25m ago', event: 'DiffDock docking completed for batch #47' },
            ].map((item, i) => (
              <div key={i} className="flex items-start space-x-3">
                <span className="text-xs text-gray-500 w-16 shrink-0">{item.time}</span>
                <span className="text-sm text-gray-300">{item.event}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
