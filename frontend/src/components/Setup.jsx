import React, { useState } from 'react';
import ThemeSwitch from './ThemeSwitch';
import { useTheme } from '../ThemeContext';
import './Setup.css';

const PRIVACY_EXAMPLE = `I believe people have a fundamental right to privacy in their personal lives and communications.

Companies should not collect, store, or sell personal data without explicit, informed consent. "Informed" means the person genuinely understands what data is being collected and how it will be used — not just clicking through a 50-page terms of service.

Surveillance by the government should require a warrant based on probable cause. Mass surveillance of citizens without individualized suspicion is wrong.

Medical, financial, and communications data deserve especially strong protection. These should never be shared without explicit consent except in genuine emergencies.

However, privacy is not absolute. When there is a specific, credible threat to someone's safety, privacy can be overridden — but only to the minimum extent necessary and with appropriate oversight.`;

export default function Setup({ onSubmit, onDemo, isLoading }) {
  const { theme } = useTheme();
  const [mode, setMode] = useState('choose'); // 'choose' | 'create'
  const [domain, setDomain] = useState('');
  const [principles, setPrinciples] = useState('');
  const [apiKey, setApiKey] = useState('');

  const canSubmit = domain.trim() && principles.trim() && apiKey.trim() && !isLoading;

  return (
    <div className="setup">
      {/* Theme switcher — top right */}
      <div className="setup__theme">
        <ThemeSwitch />
      </div>

      {/* Title */}
      <div className="setup__title-block">
        <h1 className="setup__title">Loophole</h1>
        <p className="setup__tagline">
          Stress-test your rules. AI agents attack any principles you write, finding cases where they go wrong — you settle the disputes.
        </p>
        <div className="setup__mode-tag">{theme.tagline}</div>
      </div>

      {mode === 'choose' ? (
        /* ========== Choose what to do ========== */
        <div className="setup__choices">
          {/* Primary: Demo */}
          <button
            className="choice choice--primary"
            onClick={onDemo}
            disabled={isLoading}
          >
            <div className="choice__icon">▶</div>
            <div className="choice__body">
              <div className="choice__title">Try the demo</div>
              <div className="choice__desc">
                A finished session on privacy principles — 5 cases, 4 code revisions, one still awaiting your ruling.
              </div>
              <div className="choice__meta">
                No API key needed · 10 seconds to explore
              </div>
            </div>
            <div className="choice__arrow">→</div>
          </button>

          {/* Secondary: Create */}
          <button
            className="choice choice--secondary"
            onClick={() => setMode('create')}
            disabled={isLoading}
          >
            <div className="choice__icon">+</div>
            <div className="choice__body">
              <div className="choice__title">Create your own session</div>
              <div className="choice__desc">
                Write your own principles and watch them get attacked in real time.
              </div>
              <div className="choice__meta">
                Needs Anthropic API key · ~2 min per round
              </div>
            </div>
            <div className="choice__arrow">→</div>
          </button>
        </div>
      ) : (
        /* ========== Create session form ========== */
        <div className="setup__form">
          <button className="setup__back" onClick={() => setMode('choose')}>
            ← back
          </button>

          <div className="field">
            <label htmlFor="domain">Topic</label>
            <input
              id="domain"
              type="text"
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
              placeholder="privacy, property, speech, medicine..."
              autoFocus
            />
          </div>

          <div className="field">
            <div className="field__label-row">
              <label htmlFor="principles">Your principles</label>
              <button
                type="button"
                className="field__link"
                onClick={() => { setDomain('privacy'); setPrinciples(PRIVACY_EXAMPLE); }}
              >
                use privacy example
              </button>
            </div>
            <textarea
              id="principles"
              value={principles}
              onChange={(e) => setPrinciples(e.target.value)}
              placeholder="Write what you actually believe about this topic, in plain language."
              rows={7}
            />
          </div>

          <div className="field">
            <label htmlFor="apikey">Anthropic API key</label>
            <input
              id="apikey"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-ant-..."
            />
            <div className="field__hint">Used only for this session. Not stored.</div>
          </div>

          <button
            className="setup__submit"
            disabled={!canSubmit}
            onClick={() => onSubmit({
              domain, principles, anthropic_api_key: apiKey,
              max_rounds: 10, cases_per_agent: 3,
            })}
          >
            {isLoading ? 'Starting...' : 'Start session →'}
          </button>
        </div>
      )}
    </div>
  );
}
