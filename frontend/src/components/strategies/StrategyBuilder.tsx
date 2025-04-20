import React, { useState } from 'react';
import { strategiesApi } from '../../api/strategies';
import { Strategy as ApiStrategy } from '../../types';

export interface StrategyParameter {
  name: string;
  type: 'number' | 'string' | 'boolean' | 'select';
  value: number | string | boolean;
  options?: string[];
}

interface StrategyBuilderProps {
  onSave?: (strategy: ApiStrategy) => void;
}

const parameterTypes = [
  { label: 'Number', value: 'number' },
  { label: 'String', value: 'string' },
  { label: 'Boolean', value: 'boolean' },
  { label: 'Select', value: 'select' },
];

const defaultParameters: StrategyParameter[] = [
  { name: 'Lookback Period', type: 'number', value: 14 },
  { name: 'Threshold', type: 'number', value: 0.7 },
  { name: 'Use Stop Loss', type: 'boolean', value: true },
  { name: 'Order Type', type: 'select', value: 'market', options: ['market', 'limit'] },
];

const emptyParam: StrategyParameter = { name: '', type: 'number', value: 0 };

const validateParams = (params: StrategyParameter[]) => {
  for (const param of params) {
    if (!param.name.trim()) return false;
    if (param.type === 'select' && (!param.options || param.options.length === 0)) return false;
  }
  return true;
};

const StrategyBuilder: React.FC<StrategyBuilderProps> = ({ onSave }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [parameters, setParameters] = useState<StrategyParameter[]>(defaultParameters);
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Parameter management
  const addParameter = () => setParameters([...parameters, { ...emptyParam }]);
  const removeParameter = (idx: number) => setParameters(parameters.filter((_, i) => i !== idx));
  const handleParamField = (idx: number, field: keyof StrategyParameter, value: any) => {
    setParameters(params => params.map((p, i) => i === idx ? { ...p, [field]: value, ...(field === 'type' ? { value: value === 'boolean' ? false : value === 'number' ? 0 : '', options: value === 'select' ? ['option1'] : undefined } : {}) } : p));
  };
  const handleParamOptionChange = (idx: number, options: string) => {
    setParameters(params => params.map((p, i) => i === idx ? { ...p, options: options.split(',').map(o => o.trim()).filter(Boolean) } : p));
  };

  // Validation
  const isValid = name.trim() && parameters.length > 0 && validateParams(parameters);

  // Save to backend
  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const paramObj: Record<string, any> = {};
      parameters.forEach(p => {
        paramObj[p.name] = p.type === 'select' ? { value: p.value, options: p.options } : p.value;
      });
      const payload = { name, description, parameters: paramObj };
      const saved = await strategiesApi.createStrategy(payload);
      setSuccess(true);
      if (onSave) onSave(saved);
      setTimeout(() => setSuccess(false), 2000);
      setName('');
      setDescription('');
      setParameters([]);
    } catch (e: any) {
      setError(e.message || 'Failed to save strategy');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-6 max-w-xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Strategy Builder</h2>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Strategy Name<span className="text-red-500">*</span></label>
        <input
          className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder="Enter strategy name"
        />
      </div>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Description</label>
        <textarea
          className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
          value={description}
          onChange={e => setDescription(e.target.value)}
          placeholder="Describe your strategy"
        />
      </div>
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">Parameters<span className="text-red-500">*</span></label>
        <div className="space-y-3">
          {parameters.map((param, idx) => (
            <div key={idx} className="flex items-center gap-2 mb-2">
              <input
                className="w-36 px-2 py-1 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
                value={param.name}
                onChange={e => handleParamField(idx, 'name', e.target.value)}
                placeholder="Parameter name"
              />
              <select
                className="w-28 px-2 py-1 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
                value={param.type}
                onChange={e => handleParamField(idx, 'type', e.target.value)}
              >
                {parameterTypes.map(pt => <option key={pt.value} value={pt.value}>{pt.label}</option>)}
              </select>
              {/* Value input */}
              {param.type === 'number' && (
                <input
                  type="number"
                  className="w-20 px-2 py-1 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
                  value={param.value as number}
                  onChange={e => handleParamField(idx, 'value', Number(e.target.value))}
                />
              )}
              {param.type === 'string' && (
                <input
                  type="text"
                  className="w-32 px-2 py-1 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
                  value={param.value as string}
                  onChange={e => handleParamField(idx, 'value', e.target.value)}
                />
              )}
              {param.type === 'boolean' && (
                <input
                  type="checkbox"
                  checked={!!param.value}
                  onChange={e => handleParamField(idx, 'value', e.target.checked)}
                />
              )}
              {param.type === 'select' && (
                <>
                  <input
                    type="text"
                    className="w-32 px-2 py-1 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
                    value={param.value as string}
                    onChange={e => handleParamField(idx, 'value', e.target.value)}
                    placeholder="Selected value"
                  />
                  <input
                    type="text"
                    className="w-40 px-2 py-1 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
                    value={param.options ? param.options.join(',') : ''}
                    onChange={e => handleParamOptionChange(idx, e.target.value)}
                    placeholder="Options (comma-separated)"
                  />
                </>
              )}
              <button
                className="ml-2 px-2 py-1 rounded bg-red-500 text-white hover:bg-red-600"
                onClick={() => removeParameter(idx)}
                type="button"
                title="Remove Parameter"
              >
                ×
              </button>
            </div>
          ))}
        </div>
        <button
          className="mt-2 px-4 py-1 rounded bg-blue-500 text-white hover:bg-blue-600"
          onClick={addParameter}
          type="button"
        >
          + Add Parameter
        </button>
      </div>
      <button
        className="bg-primary text-white px-6 py-2 rounded font-semibold hover:bg-primary-dark disabled:opacity-50"
        onClick={handleSave}
        disabled={saving || !isValid}
      >
        {saving ? 'Saving...' : 'Save Strategy'}
      </button>
      {success && (
        <div className="mt-4 text-green-600 font-medium">Strategy saved successfully!</div>
      )}
      {error && (
        <div className="mt-4 text-red-600 font-medium">{error}</div>
      )}
    </div>
  );
};

export default StrategyBuilder;
