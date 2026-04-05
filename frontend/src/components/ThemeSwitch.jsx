import React, { useState, useRef, useEffect } from 'react';
import { useTheme, THEMES } from '../ThemeContext';
import './ThemeSwitch.css';

export default function ThemeSwitch() {
  const { themeKey, setThemeKey, theme } = useTheme();
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    function onClick(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, []);

  return (
    <div className="theme-switch" ref={ref}>
      <button
        className="theme-switch__trigger"
        onClick={() => setOpen(!open)}
        aria-label="Switch theme"
      >
        <span className={`theme-switch__swatch theme-switch__swatch--${themeKey}`} />
        <span className="theme-switch__name">{theme.name}</span>
        <span className="theme-switch__caret">▾</span>
      </button>

      {open && (
        <div className="theme-switch__menu">
          {Object.entries(THEMES).map(([key, t]) => (
            <button
              key={key}
              className={`theme-switch__option ${key === themeKey ? 'theme-switch__option--active' : ''}`}
              onClick={() => { setThemeKey(key); setOpen(false); }}
            >
              <span className={`theme-switch__swatch theme-switch__swatch--${key}`} />
              <span className="theme-switch__info">
                <span className="theme-switch__option-name">{t.name}</span>
                <span className="theme-switch__option-desc">{t.tagline}</span>
              </span>
              {key === themeKey && <span className="theme-switch__check">✓</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
