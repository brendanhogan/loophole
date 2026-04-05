import React, { useState, useEffect, useCallback } from 'react';
import Setup from './components/Setup';
import Court from './components/Court';
import { ThemeProvider } from './ThemeContext';
import { useSession } from './hooks/useSession';
import { useWebSocket } from './hooks/useWebSocket';
import './index.css';
import './App.css';

export default function App() {
  return (
    <ThemeProvider>
      <AppInner />
    </ThemeProvider>
  );
}

function AppInner() {
  const [session, setSession] = useState(null);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const { createSession, startRound, submitDecision, submitVote, submitSuggestion, loadDemo } = useSession();
  const { wsMessages } = useWebSocket(session?.session_id);

  // Absorb live updates from WebSocket
  useEffect(() => {
    if (!wsMessages.length) return;
    const msg = wsMessages[wsMessages.length - 1];

    switch (msg.type) {
      case 'session_created':
        setSession(msg.session);
        setStatus('idle');
        setIsLoading(false);
        break;
      case 'round_started':
        setStatus('searching');
        break;
      case 'agent_working':
        setStatus(msg.agent?.includes('finder') ? 'searching' : 'judging');
        break;
      case 'judge_evaluating':
        setStatus('judging');
        break;
      case 'case_escalated':
        setStatus('escalated');
        setSession(prev => prev ? { ...prev, cases: upsertCase(prev.cases, msg.case) } : prev);
        break;
      case 'case_resolved':
      case 'user_decision_applied':
        setSession(prev => prev ? {
          ...prev,
          cases: upsertCase(prev.cases, msg.case),
          current_code: msg.new_code,
          code_history: [...prev.code_history, msg.new_code]
        } : prev);
        break;
      case 'round_complete':
        setStatus('idle');
        if (msg.session) setSession(msg.session);
        break;
      case 'error':
        setError(msg.message);
        setStatus('idle');
        setIsLoading(false);
        break;
      default: break;
    }
  }, [wsMessages]);

  const handleCreateSession = async (data) => {
    try {
      setIsLoading(true); setError(null); setStatus('drafting');
      const res = await createSession(data);
      setSession(res.state);
    } catch (err) {
      setError(err.message); setIsLoading(false); setStatus('idle');
    }
  };

  const handleLoadDemo = async () => {
    try {
      setIsLoading(true); setError(null);
      const demo = await loadDemo();
      setSession(demo);
      setStatus('idle');
      setIsLoading(false);
    } catch (err) {
      setError(err.message); setIsLoading(false);
    }
  };

  const handleStartRound = useCallback(async () => {
    if (!session) return;
    try { setError(null); await startRound(session.session_id); }
    catch (err) { setError(err.message); }
  }, [session, startRound]);

  const handleDecision = useCallback(async (caseId, decision) => {
    if (!session) return;
    try { await submitDecision(session.session_id, caseId, decision); }
    catch (err) { setError(err.message); }
  }, [session, submitDecision]);

  const handleVote = useCallback((caseId, pred) => {
    if (!session) return;
    submitVote(session.session_id, caseId, pred).catch(() => {});
  }, [session, submitVote]);

  const handleSuggestion = useCallback((caseId, sugg) => {
    if (!session) return;
    submitSuggestion(session.session_id, caseId, sugg).catch(() => {});
  }, [session, submitSuggestion]);

  const handleReset = () => { setSession(null); setStatus('idle'); setError(null); };

  return (
    <>
      {error && (
        <div className="error-banner">
          <span className="error-banner__label">Error</span>
          <span className="error-banner__msg">{error}</span>
          <button onClick={() => setError(null)}>dismiss</button>
        </div>
      )}

      {!session ? (
        <Setup
          onSubmit={handleCreateSession}
          onDemo={handleLoadDemo}
          isLoading={isLoading}
        />
      ) : (
        <Court
          session={session}
          status={status}
          onStartRound={handleStartRound}
          onDecision={handleDecision}
          onVote={handleVote}
          onSuggestion={handleSuggestion}
          onReset={handleReset}
        />
      )}
    </>
  );
}

function upsertCase(cases, newCase) {
  const idx = cases.findIndex(c => c.id === newCase.id);
  if (idx >= 0) { const next = [...cases]; next[idx] = newCase; return next; }
  return [...cases, newCase];
}
