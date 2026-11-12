import * as SecureStore from 'expo-secure-store';

export type Session = { server: string; token: string; username: string };

export type Camera = {
  id: string;
  name: string;
  location: string;
  state: string;
  people: number;
  down_ids: number[];
};

export type AlertStatus = 'active' | 'acknowledged' | 'resolved';

export type FallAlert = {
  id: number;
  camera_id: string;
  camera_name: string;
  location: string;
  people_down: number;
  raised_at: string;
  status: AlertStatus;
  escalated: boolean;
  acted_by: string | null;
  acted_at: string | null;
};

const SESSION_KEY = 'safefall-session';

export class SignedOutError extends Error {}

export async function loadSession(): Promise<Session | null> {
  const saved = await SecureStore.getItemAsync(SESSION_KEY);
  return saved ? (JSON.parse(saved) as Session) : null;
}

export function saveSession(session: Session) {
  return SecureStore.setItemAsync(SESSION_KEY, JSON.stringify(session));
}

export function clearSession() {
  return SecureStore.deleteItemAsync(SESSION_KEY);
}

export function cleanServer(address: string) {
  const trimmed = address.trim().replace(/\/+$/, '');
  return /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
}

async function readError(response: Response) {
  try {
    const body = await response.json();
    return String(body.error ?? `Server error ${response.status}`);
  } catch {
    return `Server error ${response.status}`;
  }
}

export async function signIn(address: string, username: string, password: string): Promise<Session> {
  const server = cleanServer(address);
  const response = await fetch(`${server}/api/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  const body = await response.json();
  return { server, token: body.token, username: body.username };
}

async function request<T>(session: Session, path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${session.server}/api${path}`, {
    ...init,
    headers: {
      ...authHeaders(session),
      'Content-Type': 'application/json',
    },
  });
  if (response.status === 401) {
    throw new SignedOutError(await readError(response));
  }
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

export function authHeaders(session: Session) {
  return { Authorization: `Bearer ${session.token}` };
}

export const getCameras = (session: Session) => request<Camera[]>(session, '/cameras');

export const getAlerts = (session: Session) => request<FallAlert[]>(session, '/alerts');

export const getAlert = (session: Session, id: number) =>
  request<FallAlert>(session, `/alerts/${id}`);

export const acknowledgeAlert = (session: Session, id: number) =>
  request<FallAlert>(session, `/alerts/${id}/ack`, { method: 'POST' });

export const resolveAlert = (session: Session, id: number) =>
  request<FallAlert>(session, `/alerts/${id}/resolve`, { method: 'POST' });

export const addDevice = (session: Session, token: string) =>
  request(session, '/devices', { method: 'POST', body: JSON.stringify({ token }) });

export const removeDevice = (session: Session, token: string) =>
  request(session, '/devices', { method: 'DELETE', body: JSON.stringify({ token }) });

export const cameraImageUrl = (session: Session, cameraId: string, refresh: number) =>
  `${session.server}/api/cameras/${encodeURIComponent(cameraId)}/snapshot?t=${refresh}`;

export const alertImageUrl = (session: Session, id: number) =>
  `${session.server}/api/alerts/${id}/image`;
