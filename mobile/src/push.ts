import Constants from 'expo-constants';
import * as Device from 'expo-device';
import * as Notifications from 'expo-notifications';

import { Session, addDevice } from './api';

export const FALL_CHANNEL = 'falls'; // must match channelId in the server's alerts.py

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldPlaySound: true,
    shouldSetBadge: false,
    shouldShowBanner: true,
    shouldShowList: true,
  }),
});

/** Sets this phone up for alerts. Returns its push token, or throws a reason to show the user. */
export async function registerForAlerts(session: Session): Promise<string> {
  if (!Device.isDevice) {
    throw new Error('Alerts need a real phone.');
  }
  // The channel comes first so Android 13 and newer shows the permission prompt.
  await Notifications.setNotificationChannelAsync(FALL_CHANNEL, {
    name: 'Fall alerts',
    importance: Notifications.AndroidImportance.MAX,
    sound: 'default',
    vibrationPattern: [0, 500, 250, 500, 250, 500],
    enableVibrate: true,
    lockscreenVisibility: Notifications.AndroidNotificationVisibility.PUBLIC,
  });

  let { status } = await Notifications.getPermissionsAsync();
  if (status !== 'granted') {
    ({ status } = await Notifications.requestPermissionsAsync());
  }
  if (status !== 'granted') {
    throw new Error('Notifications are turned off for SafeFall in your phone settings.');
  }

  const projectId = Constants?.expoConfig?.extra?.eas?.projectId ?? Constants?.easConfig?.projectId;
  const { data: token } = await Notifications.getExpoPushTokenAsync({ projectId });
  await addDevice(session, token);
  return token;
}

/** Reads the alert id from a tapped notification. */
export function alertIdFrom(response: Notifications.NotificationResponse): number | null {
  if (response.actionIdentifier !== Notifications.DEFAULT_ACTION_IDENTIFIER) {
    return null;
  }
  const id = response.notification.request.content.data?.alertId;
  return typeof id === 'number' ? id : null;
}
