import React from 'react';
import Legislation from './Legislation';
import Docket from './Docket';
import ThemeSwitch from './ThemeSwitch';
import { useLabels } from '../ThemeContext';
import './Court.css';

export default function Court({ session, status, onStartRound, onDecision, onVote, onSuggestion, onReset }) {
  const L = useLabels();
  const resolvedCount = session.cases.filter(c => c.status === 'auto_resolved' || c.status === 'user_resolved').length;
  const escalatedCount = session.cases.filter(c => c.status === 'escalated').length;

  return (
    <div className="court">
      {/* Top Bar — always present, structural */}
      <header className="court__header">
        <button className="court__brand" onClick={onReset} title="Return to title page">
          <span className="court__brand-mark">L</span>
          <span className="court__brand-word">oophole</span>
        </button>

        <div className="court__meta">
          <div className="court__meta-item">
            <span className="court__meta-label">topic</span>
            <span className="court__meta-value">{session.domain}</span>
          </div>
          <div className="court__meta-divider" />
          <div className="court__meta-item">
            <span className="court__meta-label">{L.round.toLowerCase()}</span>
            <span className="court__meta-value">{session.current_round}</span>
          </div>
          <div className="court__meta-divider" />
          <div className="court__meta-item">
            <span className="court__meta-label">{L.cases.toLowerCase()}</span>
            <span className="court__meta-value">{session.cases.length}</span>
          </div>
        </div>

        <div className="court__status">
          <StatusPulse status={status} L={L} />
          <ThemeSwitch />
        </div>
      </header>

      {/* Summary bar */}
      <div className="court__ticker">
        <div className="court__ticker-scroll">
          <span><strong>{resolvedCount}</strong> resolved</span>
          <span className="court__ticker-dot">·</span>
          <span><strong>{escalatedCount}</strong> {L.needsYouShort ? L.needsYouShort.toLowerCase() : 'need your ruling'}</span>
          <span className="court__ticker-dot">·</span>
          <span><strong>{session.code_history.length}</strong> {L.codeShort.toLowerCase()} revisions</span>
        </div>
      </div>

      {/* Main split */}
      <div className="court__body">
        <section className="court__panel court__panel--left">
          <Legislation session={session} />
        </section>

        <div className="court__divider">
          <div className="court__divider-line" />
          <div className="court__divider-seal">§</div>
          <div className="court__divider-line" />
        </div>

        <section className="court__panel court__panel--right">
          <Docket
            session={session}
            status={status}
            onStartRound={onStartRound}
            onDecision={onDecision}
            onVote={onVote}
            onSuggestion={onSuggestion}
          />
        </section>
      </div>
    </div>
  );
}

function StatusPulse({ status, L }) {
  const labels = {
    idle: L.ready,
    drafting: L.drafting,
    searching: L.searching,
    judging: L.judging,
    escalated: L.escalated,
  };
  const colors = {
    idle: 'var(--ink-faint)',
    drafting: 'var(--accent)',
    searching: 'var(--loophole)',
    judging: 'var(--overreach)',
    escalated: 'var(--signal)',
  };
  return (
    <div className="status-pulse">
      <span
        className={`status-pulse__dot ${status !== 'idle' ? 'status-pulse__dot--live' : ''}`}
        style={{ backgroundColor: colors[status] || colors.idle }}
      />
      <span className="status-pulse__label">{labels[status] || labels.idle}</span>
    </div>
  );
}
