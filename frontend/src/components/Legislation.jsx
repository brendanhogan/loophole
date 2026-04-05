import React, { useState, useRef, useEffect } from 'react';
import { useLabels } from '../ThemeContext';
import './Legislation.css';

export default function Legislation({ session }) {
  const L = useLabels();
  const [view, setView] = useState('current'); // 'current' | 'history'
  const [selectedVersion, setSelectedVersion] = useState(null);
  const codeRef = useRef(null);

  const currentCode = session.current_code;
  const history = session.code_history || [];

  // Scroll code to top when version changes
  useEffect(() => {
    codeRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  }, [currentCode.version]);

  const displayedCode = selectedVersion !== null
    ? history.find(h => h.version === selectedVersion) || currentCode
    : currentCode;

  return (
    <div className="legislation">
      {/* Head — the code's running head */}
      <header className="legislation__head">
        <div className="legislation__head-left">
          <h2 className="legislation__title">
            {L.panelTitle} · {capitalize(session.domain)}
          </h2>
          <p className="legislation__subtitle">{L.panelSubtitle}</p>
        </div>
        <div className="legislation__head-right">
          <div className="legislation__version">
            <span className="legislation__version-label">version</span>
            <span className="legislation__version-num">{displayedCode.version}</span>
          </div>
        </div>
      </header>

      {/* View toggle */}
      <nav className="legislation__tabs">
        <button
          className={`legislation__tab ${view === 'current' ? 'legislation__tab--active' : ''}`}
          onClick={() => { setView('current'); setSelectedVersion(null); }}
        >
          Current
        </button>
        <button
          className={`legislation__tab ${view === 'history' ? 'legislation__tab--active' : ''}`}
          onClick={() => setView('history')}
        >
          History
          <span className="legislation__tab-count">{history.length}</span>
        </button>
      </nav>

      {/* Content */}
      {view === 'current' ? (
        <div className="legislation__content" ref={codeRef}>
          {displayedCode.changelog && displayedCode.version > 1 && (
            <div className="legislation__changelog">
              <div className="legislation__changelog-meta">
                Latest change · {new Date(displayedCode.created_at).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}
              </div>
              <p>{displayedCode.changelog}</p>
            </div>
          )}

          <CodeText text={displayedCode.text} />

          <footer className="legislation__footer">
            <div className="legislation__footer-mark">— fin. —</div>
          </footer>
        </div>
      ) : (
        <div className="legislation__content">
          <div className="legislation__history">
            {history.slice().reverse().map((v, idx) => {
              const isLatest = idx === 0;
              return (
                <article
                  key={v.version}
                  className={`amendment ${isLatest ? 'amendment--latest' : ''}`}
                  onClick={() => { setSelectedVersion(v.version); setView('current'); }}
                >
                  <div className="amendment__gutter">
                    <span className="amendment__num">{v.version}</span>
                    {!isLatest && <div className="amendment__line" />}
                  </div>
                  <div className="amendment__body">
                    <div className="amendment__meta">
                      <span className="amendment__label">
                        {v.version === 1 ? 'Original draft' : `Amendment ${v.version - 1}`}
                      </span>
                      <span className="amendment__time">
                        {new Date(v.created_at).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}
                      </span>
                    </div>
                    <p className="amendment__changelog">
                      {v.changelog || <em>Original codification from stated principles</em>}
                    </p>
                  </div>
                </article>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

/* ----- Code rendering: parse ARTICLE / § structure ---------------------- */
function CodeText({ text }) {
  const lines = text.split('\n');
  const blocks = [];
  let current = null;

  for (const line of lines) {
    if (/^ARTICLE\s+/i.test(line)) {
      if (current) blocks.push(current);
      current = { type: 'article', title: line, sections: [] };
    } else if (/^§/.test(line.trim())) {
      if (!current) current = { type: 'article', title: '', sections: [] };
      current.sections.push(line.trim());
    } else if (line.trim() === '') {
      // skip blank lines; they separate articles
    } else if (current && current.sections.length > 0) {
      // continuation of previous section
      current.sections[current.sections.length - 1] += ' ' + line.trim();
    } else if (current) {
      current.title += ' ' + line.trim();
    }
  }
  if (current) blocks.push(current);

  // Fallback: if no structure, just show raw
  if (blocks.length === 0) {
    return <pre className="code-text__raw">{text}</pre>;
  }

  return (
    <div className="code-text">
      {blocks.map((block, i) => (
        <section key={i} className="code-text__article">
          <h3 className="code-text__article-title">{block.title}</h3>
          <ol className="code-text__sections">
            {block.sections.map((sec, j) => {
              const isNew = /\[NEW\]/.test(sec);
              const cleaned = sec.replace(/\[NEW\]\s*/, '');
              const [marker, ...rest] = cleaned.split(' ');
              const body = rest.join(' ');
              return (
                <li key={j} className={`code-text__section ${isNew ? 'code-text__section--new' : ''}`}>
                  <span className="code-text__marker">{marker}</span>
                  <span className="code-text__body">{body}</span>
                  {isNew && <span className="code-text__badge">new</span>}
                </li>
              );
            })}
          </ol>
        </section>
      ))}
    </div>
  );
}

function capitalize(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : ''; }
