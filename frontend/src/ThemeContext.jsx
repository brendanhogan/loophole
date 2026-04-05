import React, { createContext, useContext, useState, useEffect } from 'react';
import { THEMES, DEFAULT_THEME, getTheme } from './themes';

const ThemeContext = createContext(null);
const STORAGE_KEY = 'loophole-theme';

export function ThemeProvider({ children }) {
  const [themeKey, setThemeKey] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) || DEFAULT_THEME;
    } catch { return DEFAULT_THEME; }
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', themeKey);
    try { localStorage.setItem(STORAGE_KEY, themeKey); } catch {}
  }, [themeKey]);

  const theme = getTheme(themeKey);

  return (
    <ThemeContext.Provider value={{ themeKey, setThemeKey, theme, labels: theme.labels }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider');
  return ctx;
}

export function useLabels() {
  return useTheme().labels;
}

export { THEMES };
