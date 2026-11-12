import { useCallback, useState } from 'react';
import { Image } from 'expo-image';
import {
  ActivityIndicator,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';

import {
  Camera,
  FallAlert,
  Session,
  acknowledgeAlert,
  alertImageUrl,
  authHeaders,
  cameraImageUrl,
  getAlert,
  getAlerts,
  getCameras,
  resolveAlert,
  signIn,
} from './api';
import { Banner, StatusTag, colors, formatTime, useApp, useLiveData } from './ui';

export type Screens = {
  Home: undefined;
  Camera: { camera: Camera };
  Alert: { id: number };
};

const LIST_REFRESH_MS = 3000;
const NEXT_PICTURE_MS = 250;
const PICTURE_RETRY_MS = 2000;

export function SignInScreen({ onSignedIn }: { onSignedIn: (session: Session) => void }) {
  const [server, setServer] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      onSignedIn(await signIn(server, username, password));
    } catch (problem) {
      setError(problem instanceof TypeError
        ? "Can't reach that server. Check the address."
        : (problem as Error).message);
      setBusy(false);
    }
  }

  return (
    <KeyboardAvoidingView
      style={styles.signIn}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <Text style={styles.brand}>SafeFall</Text>
      <Text style={styles.muted}>Sign in to monitor your cameras</Text>
      {error && <Banner text={error} />}
      <Text style={styles.label}>Server address</Text>
      <TextInput
        style={styles.input}
        value={server}
        onChangeText={setServer}
        placeholder="https://your-address.trycloudflare.com"
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
      />
      <Text style={styles.label}>Username</Text>
      <TextInput
        style={styles.input}
        value={username}
        onChangeText={setUsername}
        autoCapitalize="none"
        autoCorrect={false}
      />
      <Text style={styles.label}>Password</Text>
      <TextInput
        style={styles.input}
        value={password}
        onChangeText={setPassword}
        secureTextEntry
        onSubmitEditing={submit}
      />
      <Pressable
        style={[styles.button, (busy || !server || !username || !password) && styles.disabled]}
        onPress={submit}
        disabled={busy || !server || !username || !password}
      >
        {busy ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>Sign in</Text>}
      </Pressable>
    </KeyboardAvoidingView>
  );
}

export function CamerasScreen({ onOpen }: { onOpen: (camera: Camera) => void }) {
  const { session, signOut, pushProblem } = useApp();
  const load = useCallback(() => getCameras(session), [session]);
  const { data, error } = useLiveData(load, LIST_REFRESH_MS);

  return (
    <FlatList
      style={styles.screen}
      contentContainerStyle={styles.list}
      data={data ?? []}
      keyExtractor={camera => camera.id}
      ListHeaderComponent={
        <>
          {pushProblem && <Banner tone="warning" text={`Alerts are off: ${pushProblem}`} />}
          {error && <Banner text={error} />}
        </>
      }
      ListEmptyComponent={data ? <Text style={styles.muted}>No cameras are set up.</Text> : <ActivityIndicator />}
      ListFooterComponent={
        <Pressable onPress={signOut} style={styles.signOut}>
          <Text style={styles.link}>Sign out of {session.username}</Text>
        </Pressable>
      }
      renderItem={({ item }) => {
        const down = item.down_ids.length;
        return (
          <Pressable style={[styles.card, down > 0 && styles.cardDanger]} onPress={() => onOpen(item)}>
            <Text style={styles.title}>{item.name}</Text>
            <Text style={styles.muted}>{item.location}</Text>
            <Text style={down > 0 ? styles.dangerText : styles.muted}>
              {down > 0
                ? `${down} ${down === 1 ? 'person' : 'people'} down`
                : item.state === 'running'
                  ? `${item.people} in view`
                  : `Camera ${item.state}`}
            </Text>
          </Pressable>
        );
      }}
    />
  );
}

export function CameraScreen({ route }: NativeStackScreenProps<Screens, 'Camera'>) {
  const { session } = useApp();
  const { camera } = route.params;
  const [refresh, setRefresh] = useState(0);
  const [failed, setFailed] = useState(false);

  // The next picture is requested only after this one arrives, so slow networks don't queue up.
  const next = useCallback((delay: number) => {
    setTimeout(() => setRefresh(value => value + 1), delay);
  }, []);

  return (
    <View style={styles.screen}>
      <Image
        style={styles.picture}
        source={{ uri: cameraImageUrl(session, camera.id, refresh), headers: authHeaders(session) }}
        cachePolicy="none"
        contentFit="contain"
        onLoad={() => {
          setFailed(false);
          next(NEXT_PICTURE_MS);
        }}
        onError={() => {
          setFailed(true);
          next(PICTURE_RETRY_MS);
        }}
      />
      <View style={styles.list}>
        {failed && <Banner text="Live view isn't available right now. Trying again." />}
        <Text style={styles.title}>{camera.name}</Text>
        <Text style={styles.muted}>{camera.location}</Text>
      </View>
    </View>
  );
}

export function AlertsScreen({ onOpen }: { onOpen: (id: number) => void }) {
  const { session } = useApp();
  const load = useCallback(() => getAlerts(session), [session]);
  const { data, error } = useLiveData(load, LIST_REFRESH_MS);

  return (
    <FlatList
      style={styles.screen}
      contentContainerStyle={styles.list}
      data={data ?? []}
      keyExtractor={alert => String(alert.id)}
      ListHeaderComponent={error ? <Banner text={error} /> : undefined}
      ListEmptyComponent={data ? <Text style={styles.muted}>No alerts yet.</Text> : <ActivityIndicator />}
      renderItem={({ item }) => (
        <Pressable style={styles.card} onPress={() => onOpen(item.id)}>
          <View style={styles.row}>
            <Text style={[styles.title, styles.grow]}>{item.location}</Text>
            <StatusTag status={item.status} />
          </View>
          <Text style={styles.muted}>{item.camera_name}</Text>
          <Text style={styles.muted}>
            {formatTime(item.raised_at)} · {item.people_down} down
          </Text>
        </Pressable>
      )}
    />
  );
}

export function AlertScreen({ route }: NativeStackScreenProps<Screens, 'Alert'>) {
  const { session } = useApp();
  const { id } = route.params;
  const load = useCallback(() => getAlert(session, id), [session, id]);
  const { data: alert, error, setData } = useLiveData<FallAlert>(load, LIST_REFRESH_MS);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  async function act(action: typeof acknowledgeAlert) {
    setBusy(true);
    setActionError(null);
    try {
      setData(await action(session, id));
    } catch (problem) {
      setActionError((problem as Error).message);
    }
    setBusy(false);
  }

  if (!alert) {
    return (
      <View style={[styles.screen, styles.list]}>
        {error ? <Banner text={error} /> : <ActivityIndicator />}
      </View>
    );
  }

  return (
    <ScrollView style={styles.screen}>
      <Image
        style={styles.picture}
        source={{ uri: alertImageUrl(session, id), headers: authHeaders(session) }}
        contentFit="contain"
      />
      <View style={styles.list}>
        {error && <Banner text={error} />}
        {actionError && <Banner text={actionError} />}
        <StatusTag status={alert.status} />
        <Text style={styles.heading}>{alert.location}</Text>
        <Text style={styles.muted}>{alert.camera_name}</Text>
        <Text style={styles.muted}>{formatTime(alert.raised_at)}</Text>
        <Text style={styles.muted}>People down: {alert.people_down}</Text>
        {alert.escalated && <Text style={styles.dangerText}>Still down, no response yet</Text>}
        {alert.acted_by && alert.acted_at && (
          <Text style={styles.muted}>
            {alert.status === 'resolved' ? 'Resolved' : 'Acknowledged'} by {alert.acted_by},{' '}
            {formatTime(alert.acted_at)}
          </Text>
        )}
        {alert.status === 'active' && (
          <Pressable
            style={[styles.button, styles.dangerButton, busy && styles.disabled]}
            disabled={busy}
            onPress={() => act(acknowledgeAlert)}
          >
            <Text style={styles.buttonText}>I'm responding</Text>
          </Pressable>
        )}
        {alert.status !== 'resolved' && (
          <Pressable
            style={[styles.button, styles.secondaryButton, busy && styles.disabled]}
            disabled={busy}
            onPress={() => act(resolveAlert)}
          >
            <Text style={styles.link}>Mark as resolved</Text>
          </Pressable>
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background },
  list: { padding: 16, gap: 10 },
  signIn: { flex: 1, justifyContent: 'center', padding: 24, gap: 8, backgroundColor: colors.background },
  brand: { fontSize: 32, fontWeight: '700', color: colors.text },
  label: { marginTop: 8, fontWeight: '600', color: colors.text },
  input: {
    backgroundColor: colors.card,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: 8,
    padding: 12,
    color: colors.text,
  },
  button: {
    backgroundColor: colors.primary,
    borderRadius: 8,
    padding: 14,
    alignItems: 'center',
    marginTop: 12,
  },
  dangerButton: { backgroundColor: colors.danger },
  secondaryButton: { backgroundColor: colors.card, borderColor: colors.border, borderWidth: 1 },
  disabled: { opacity: 0.5 },
  buttonText: { color: '#fff', fontWeight: '600', fontSize: 16 },
  card: {
    backgroundColor: colors.card,
    borderRadius: 10,
    padding: 14,
    gap: 3,
    borderColor: colors.border,
    borderWidth: 1,
  },
  cardDanger: { borderColor: colors.danger, backgroundColor: colors.dangerSoft },
  row: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  grow: { flex: 1 },
  title: { fontSize: 16, fontWeight: '600', color: colors.text },
  heading: { fontSize: 22, fontWeight: '700', color: colors.text },
  muted: { color: colors.muted },
  dangerText: { color: colors.danger, fontWeight: '600' },
  link: { color: colors.primary, fontWeight: '600' },
  signOut: { padding: 16, alignItems: 'center' },
  picture: { width: '100%', aspectRatio: 4 / 3, backgroundColor: '#000' },
});
