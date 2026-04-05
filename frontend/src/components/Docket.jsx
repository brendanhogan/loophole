import React, { useState, useRef, useEffect } from 'react';
import { useLabels } from '../ThemeContext';
import './Docket.css';

export default function Docket({ session, status, onStartRound, onDecision, onVote, onSuggestion }) {
  const L = useLabels();
  const [filter, setFilter] = useState('all'); // all | loopholes | overreach | escalated
  const endRef = useRef(null);
  const cases = session.cases || [];

  const filtered = cases.filter(c => {
    if (filter === 'all') return true;
    if (filter === 'loopholes') return c.case_type === 'loophole';
    if (filter === 'overreach') return c.case_type === 'overreach';
    if (filter === 'escalated') return c.status === 'escalated';
    return true;
  });

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [cases.length]);

  return (
    <div className="docket">
      <header className="docket__head">
        <div className="docket__head-left">
          <h2 className="docket__title">{L.docketTitle}</h2>
          <p className="docket__subtitle">{L.docketSubtitle}</p>
        </div>
        <button
          className="docket__begin"
          onClick={onStartRound}
          disabled={status !== 'idle'}
        >
          {status === 'idle' ? (
            <><span>{L.runRound}</span> <span className="docket__begin-arrow">→</span></>
          ) : (
            <span className="docket__begin-working">{L.running}</span>
          )}
        </button>
      </header>

      {/* Filters */}
      <nav className="docket__filters">
        {[
          { key: 'all', label: 'All', count: cases.length },
          { key: 'loopholes', label: 'Loopholes', count: cases.filter(c => c.case_type === 'loophole').length },
          { key: 'overreach', label: 'Overreach', count: cases.filter(c => c.case_type === 'overreach').length },
          { key: 'escalated', label: L.needsYou, count: cases.filter(c => c.status === 'escalated').length },
        ].map(f => (
          <button
            key={f.key}
            className={`docket__filter ${filter === f.key ? 'docket__filter--active' : ''}`}
            onClick={() => setFilter(f.key)}
          >
            <span>{f.label}</span>
            <span className="docket__filter-count">{f.count}</span>
          </button>
        ))}
      </nav>

      <div className="docket__scroll">
        {filtered.length === 0 ? (
          <EmptyState status={status} onStartRound={onStartRound} L={L} />
        ) : (
          <ol className="docket__list">
            {filtered.map((c, i) => (
              <Exhibit
                key={c.id}
                caseObj={c}
                index={i}
                L={L}
                onDecision={onDecision}
                onVote={onVote}
                onSuggestion={onSuggestion}
              />
            ))}
          </ol>
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}

/* ==========================================================================
   Exhibit — a single case card
   ========================================================================== */
function Exhibit({ caseObj, index, L, onDecision, onVote, onSuggestion }) {
  const [decision, setDecision] = useState('');
  const [suggestion, setSuggestion] = useState('');
  const [showSuggestion, setShowSuggestion] = useState(false);
  const [userVote, setUserVote] = useState(null);
  const [expanded, setExpanded] = useState(true);

  const isLoophole = caseObj.case_type === 'loophole';
  const isEscalated = caseObj.status === 'escalated';
  const isResolved = caseObj.status === 'auto_resolved' || caseObj.status === 'user_resolved';
  const isPending = caseObj.status === 'pending';

  const submitVote = (v) => {
    if (isEscalated || isResolved) return;
    setUserVote(v);
    onVote?.(caseObj.id, v);
  };

  const submitDecision = () => {
    if (!decision.trim()) return;
    onDecision?.(caseObj.id, decision);
    setDecision('');
  };

  const submitSuggestion = () => {
    if (!suggestion.trim()) return;
    onSuggestion?.(caseObj.id, suggestion);
    setSuggestion('');
    setShowSuggestion(false);
  };

  return (
    <li
      className={`exhibit exhibit--${caseObj.case_type} ${isEscalated ? 'exhibit--escalated' : ''} ${isResolved ? 'exhibit--resolved' : ''}`}
      style={{ animationDelay: `${Math.min(index * 60, 400)}ms` }}
    >
      <button className="exhibit__head" onClick={() => setExpanded(!expanded)}>
        <div className="exhibit__head-left">
          <div className="exhibit__number">
            <span className="exhibit__number-label">Exhibit</span>
            <span className="exhibit__number-val">{String(caseObj.id).padStart(2, '0')}</span>
          </div>
          <div className="exhibit__classification">
            <span className={`exhibit__type exhibit__type--${caseObj.case_type}`}>
              {isLoophole ? 'Loophole' : 'Overreach'}
            </span>
            <span className="exhibit__subtype">
              {isLoophole ? 'legal · immoral' : 'illegal · moral'}
            </span>
          </div>
        </div>
        <div className="exhibit__head-right">
          <StatusBadge status={caseObj.status} />
          <span className={`exhibit__chevron ${expanded ? 'exhibit__chevron--open' : ''}`}>▸</span>
        </div>
      </button>

      {expanded && (
        <div className="exhibit__body">
          <section className="exhibit__section">
            <div className="exhibit__section-label">Scenario</div>
            <p className="exhibit__scenario">{caseObj.scenario}</p>
          </section>

          <section className="exhibit__section">
            <div className="exhibit__section-label">Why it's a problem</div>
            <p className="exhibit__problem">{caseObj.explanation}</p>
          </section>

          {isResolved && caseObj.resolution && (
            <section className="exhibit__section exhibit__section--resolution">
              <div className="exhibit__section-label">
                {caseObj.resolved_by === 'user' ? L.yourRuling : L.resolution}
              </div>
              <p className="exhibit__resolution">{caseObj.resolution}</p>
              <div className="exhibit__resolved-by">
                {caseObj.resolved_by === 'user' ? L.ruledBy : L.autoResolved}
              </div>
            </section>
          )}

          {isEscalated && (
            <section className="exhibit__section exhibit__escalation">
              <div className="exhibit__escalation-banner">
                <strong>{L.escalationBanner}</strong>
                <span>{L.escalationReason}</span>
              </div>

              <textarea
                className="exhibit__decision"
                value={decision}
                onChange={(e) => setDecision(e.target.value)}
                placeholder={L.rulingPlaceholder}
                rows={4}
              />
              <div className="exhibit__decision-actions">
                <button
                  className="exhibit__submit"
                  onClick={submitDecision}
                  disabled={!decision.trim()}
                >
                  {L.submitRuling} →
                </button>
              </div>
            </section>
          )}

          {isPending && (
            <section className="exhibit__interactive">
              <div className="exhibit__predict">
                <span className="exhibit__predict-label">Your guess:</span>
                <button
                  className={`exhibit__vote ${userVote === false ? 'exhibit__vote--active' : ''}`}
                  onClick={() => submitVote(false)}
                >
                  {L.judgeWillFix}
                </button>
                <button
                  className={`exhibit__vote ${userVote === true ? 'exhibit__vote--active' : ''}`}
                  onClick={() => submitVote(true)}
                >
                  {L.needsYou}
                </button>
              </div>

              {!showSuggestion ? (
                <button className="exhibit__suggest-trigger" onClick={() => setShowSuggestion(true)}>
                  {L.suggestFix}
                </button>
              ) : (
                <div className="exhibit__suggest">
                  <textarea
                    value={suggestion}
                    onChange={(e) => setSuggestion(e.target.value)}
                    placeholder={L.suggestPlaceholder}
                    rows={2}
                  />
                  <div className="exhibit__suggest-actions">
                    <button onClick={() => setShowSuggestion(false)}>cancel</button>
                    <button className="exhibit__submit" onClick={submitSuggestion} disabled={!suggestion.trim()}>
                      submit →
                    </button>
                  </div>
                </div>
              )}
            </section>
          )}
        </div>
      )}
    </li>
  );
}

function StatusBadge({ status }) {
  const map = {
    pending:        { label: 'pending', cls: 'pending' },
    auto_resolved:  { label: 'resolved', cls: 'resolved' },
    user_resolved:  { label: 'ruled', cls: 'resolved' },
    escalated:      { label: 'escalated', cls: 'escalated' },
  };
  const info = map[status] || map.pending;
  return <span className={`status-badge status-badge--${info.cls}`}>{info.label}</span>;
}

function EmptyState({ status, onStartRound, L }) {
  return (
    <div className="docket__empty">
      <div className="docket__empty-title">{L.emptyTitle}</div>
      <p>{L.emptyDesc}</p>
      {status === 'idle' && (
        <button className="docket__empty-cta" onClick={onStartRound}>
          {L.runFirst} →
        </button>
      )}
    </div>
  );
}
