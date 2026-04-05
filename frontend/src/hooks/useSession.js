import { useState, useCallback } from 'react';

const API_BASE = process.env.REACT_APP_API_BASE || (
  window.location.hostname === 'localhost'
    ? 'http://localhost:8001/api'
    : '/api'
);

export const useSession = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const apiCall = useCallback(async (endpoint, options = {}) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const createSession = useCallback(async (sessionData) => {
    return apiCall('/sessions', {
      method: 'POST',
      body: JSON.stringify({
        domain: sessionData.domain,
        principles: sessionData.principles,
        anthropic_api_key: sessionData.anthropic_api_key,
        max_rounds: sessionData.max_rounds || 10,
        cases_per_agent: sessionData.cases_per_agent || 3
      })
    });
  }, [apiCall]);

  const getSession = useCallback(async (sessionId) => {
    return apiCall(`/sessions/${sessionId}`);
  }, [apiCall]);

  const startRound = useCallback(async (sessionId) => {
    return apiCall(`/sessions/${sessionId}/round`, {
      method: 'POST'
    });
  }, [apiCall]);

  const submitDecision = useCallback(async (sessionId, caseId, decision) => {
    return apiCall(`/sessions/${sessionId}/decisions`, {
      method: 'POST',
      body: JSON.stringify({
        case_id: caseId,
        decision: decision
      })
    });
  }, [apiCall]);

  const submitVote = useCallback(async (sessionId, caseId, predictedEscalation) => {
    return apiCall(`/sessions/${sessionId}/votes`, {
      method: 'POST',
      body: JSON.stringify({
        case_id: caseId,
        predicted_escalation: predictedEscalation
      })
    });
  }, [apiCall]);

  const submitSuggestion = useCallback(async (sessionId, caseId, suggestedFix) => {
    return apiCall(`/sessions/${sessionId}/suggestions`, {
      method: 'POST',
      body: JSON.stringify({
        case_id: caseId,
        suggested_fix: suggestedFix
      })
    });
  }, [apiCall]);

  const loadDemo = useCallback(async () => {
    return apiCall('/demo');
  }, [apiCall]);

  return {
    loading,
    error,
    createSession,
    getSession,
    startRound,
    submitDecision,
    submitVote,
    submitSuggestion,
    loadDemo,
  };
};