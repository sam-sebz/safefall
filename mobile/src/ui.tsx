import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import { useFocusEffect } from '@react-navigation/native';
import { StyleSheet, Text, View } from 'react-native';

import { AlertStatus, Session, SignedOutError } from './api';

export const colors = {
  background: '#f4f5f7',
  card: '#ffffff',
  text: '#16202c',
  muted: '#5c6875',
  border: '#dde1e6',
  primary: '#1f5fbf',
  danger: '#c62828',
  dangerSoft: '#fdecec',
  warning: '#9a6200',
  warningSoft: '#fdf3df',
  good: '#22703a',
  goodSoft: '#e6f4ea',
};

type AppContext = {
  session: Session;
  signOut: () => void;
  pushProblem: string | null;
};

export const AppSession = createContext<AppContext | null>(null);

export function useApp() {
  const app = useContext(AppSession);
  if (!app) {
    throw new Error('useApp is only available after sign-in');
  }
  return app;
}

/** Loads data while a screen is open, refreshing every few seconds. */
export function useLiveData<T>(load: () => Promise<T>, everyMs: number) {
  const { signOut } = useApp();
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const latestLoad = useRef(load);
  useEffect(() => {
    latestLoad.current = load;
  });

  const refresh = useCallback(async () => {
    try {
      setData(await latestLoad.current());
      setError(null);
    } catch (problem) {
      if (problem instanceof SignedOutError) {
        signOut();
      } else {
        setError("Can't reach the SafeFall server. Check that the laptop and tunnel are running.");
      }
    }
  }, [signOut]);

  useFocusEffect(
    useCallback(() => {
      refresh();
      const timer = setInterval(refresh, everyMs);
      return () => clearInterval(timer);
    }, [refresh, everyMs]),
  );

  return { data, error, refresh, setData };
}

export function Banner({ text, tone = 'danger' }: { text: string; tone?: 'danger' | 'warning' }) {
  const background = tone === 'danger' ? colors.dangerSoft : colors.warningSoft;
  const color = tone === 'danger' ? colors.danger : colors.warning;
  return (
    <View style={[styles.banner, { backgroundColor: background }]}>
      <Text style={{ color }}>{text}</Text>
    </View>
  );
}

const STATUS = {
  active: { label: 'Needs response', color: colors.danger, background: colors.dangerSoft },
  acknowledged: { label: 'Acknowledged', color: colors.warning, background: colors.warningSoft },
  resolved: { label: 'Resolved', color: colors.good, background: colors.goodSoft },
};

export function StatusTag({ status }: { status: AlertStatus }) {
  const { label, color, background } = STATUS[status];
  return (
    <View style={[styles.tag, { backgroundColor: background }]}>
      <Text style={[styles.tagText, { color }]}>{label}</Text>
    </View>
  );
}

export function formatTime(value: string) {
  return new Date(value).toLocaleString([], {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  });
}

const styles = StyleSheet.create({
  banner: { padding: 12, borderRadius: 8, marginBottom: 12 },
  tag: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6, alignSelf: 'flex-start' },
  tagText: { fontSize: 12, fontWeight: '600' },
});
