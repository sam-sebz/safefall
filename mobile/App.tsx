import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ActivityIndicator, View } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import * as Notifications from 'expo-notifications';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { NavigationContainer, createNavigationContainerRef } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { Session, clearSession, loadSession, removeDevice, saveSession } from './src/api';
import { alertIdFrom, registerForAlerts } from './src/push';
import {
  AlertScreen,
  AlertsScreen,
  CameraScreen,
  CamerasScreen,
  Screens,
  SignInScreen,
} from './src/screens';
import { AppSession, colors } from './src/ui';

const Stack = createNativeStackNavigator<Screens>();
const Tabs = createBottomTabNavigator();
const navigation = createNavigationContainerRef<Screens>();

function Home() {
  return (
    <Tabs.Navigator screenOptions={{ tabBarIconStyle: { display: 'none' }, tabBarActiveTintColor: colors.primary }}>
      <Tabs.Screen name="Cameras">
        {() => <CamerasScreen onOpen={camera => navigation.navigate('Camera', { camera })} />}
      </Tabs.Screen>
      <Tabs.Screen name="Alerts">
        {() => <AlertsScreen onOpen={id => navigation.navigate('Alert', { id })} />}
      </Tabs.Screen>
    </Tabs.Navigator>
  );
}

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const [pushProblem, setPushProblem] = useState<string | null>(null);
  const [navigationReady, setNavigationReady] = useState(false);
  const pushToken = useRef<string | null>(null);
  const openedResponse = useRef<string | null>(null);
  const tapped = Notifications.useLastNotificationResponse();

  useEffect(() => {
    loadSession()
      .then(setSession)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!session) {
      return;
    }
    registerForAlerts(session)
      .then(token => {
        pushToken.current = token;
        setPushProblem(null);
      })
      .catch((problem: Error) => setPushProblem(problem.message));
  }, [session]);

  // Covers both a running app and one started by tapping the notification.
  useEffect(() => {
    if (!tapped || !session || !navigationReady) {
      return;
    }
    const responseId = tapped.notification.request.identifier;
    const alertId = alertIdFrom(tapped);
    if (alertId !== null && openedResponse.current !== responseId) {
      openedResponse.current = responseId;
      navigation.navigate('Alert', { id: alertId });
    }
  }, [tapped, session, navigationReady]);

  const signedIn = useCallback((next: Session) => {
    saveSession(next);
    setSession(next);
  }, []);

  const signOut = useCallback(() => {
    setSession(current => {
      if (current && pushToken.current) {
        removeDevice(current, pushToken.current).catch(() => undefined);
      }
      return null;
    });
    pushToken.current = null;
    setNavigationReady(false);
    clearSession();
  }, []);

  const app = useMemo(
    () => (session ? { session, signOut, pushProblem } : null),
    [session, signOut, pushProblem],
  );

  if (loading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center' }}>
        <ActivityIndicator />
      </View>
    );
  }

  return (
    <SafeAreaProvider>
      <StatusBar style="dark" />
      {app ? (
        <AppSession.Provider value={app}>
          <NavigationContainer ref={navigation} onReady={() => setNavigationReady(true)}>
            <Stack.Navigator>
              <Stack.Screen name="Home" component={Home} options={{ headerShown: false }} />
              <Stack.Screen name="Camera" component={CameraScreen} options={{ title: 'Live view' }} />
              <Stack.Screen name="Alert" component={AlertScreen} options={{ title: 'Fall alert' }} />
            </Stack.Navigator>
          </NavigationContainer>
        </AppSession.Provider>
      ) : (
        <SignInScreen onSignedIn={signedIn} />
      )}
    </SafeAreaProvider>
  );
}
