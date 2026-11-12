# Install the SafeFall phone app

These steps haven't been run yet, because they need your own accounts and phone. If a screen looks different or a command fails, send me what you see.

Run the commands in PowerShell.

## 1. Move the app to a simple folder

The `&` in the project folder's name breaks `npx` and other Windows commands, so build from a folder without one.

```powershell
robocopy "D:\IVP PROJECT\19. SafeFall_ AI-Powered Fall Detection & Alert System with Python & Computer Vision\mobile" D:\SafeFallApp /E /XD node_modules .expo
cd D:\SafeFallApp
npm install
npm install -g eas-cli
```

## 2. Create the two free accounts

1. Sign up at [expo.dev](https://expo.dev).
2. Open the [Firebase console](https://console.firebase.google.com) and create a project called SafeFall. You can turn Google Analytics off.
3. In that project, add an **Android app** with the package name `com.safefall.monitor`.
4. Download `google-services.json` and put it in `D:\SafeFallApp`.

## 3. Connect the app to Expo

```powershell
eas login
eas init
```

Then open `D:\SafeFallApp\app.json` and add this line inside the `"android"` section:

```json
"googleServicesFile": "./google-services.json",
```

## 4. Allow Expo to send push alerts

1. In Firebase, go to **Project settings > Service accounts**.
2. Click **Generate new private key**, then **Generate key**. Save the file in `D:\SafeFallApp`. Treat it like a password and never share it.
3. Run:

```powershell
eas credentials
```

4. Choose **Android > production > Google Service Account > Manage your Google Service Account Key for Push Notifications (FCM V1) > Set up a Google Service Account Key for Push Notifications (FCM V1) > Upload a new service account key**. Press Y when it finds the file.

## 5. Build and install the APK

This folder isn't a git repository, so tell EAS to skip that check:

```powershell
$env:EAS_NO_VCS = '1'; npx --yes eas-cli@latest build -p android --profile preview
```

If your terminal says `eas` isn't recognised in any step above, use `npx --yes eas-cli@latest` in its place.

The build runs in Expo's cloud and can take a while. When it finishes, open the link it gives you on your phone, download the APK and install it. Android may ask you to allow installs from your browser.

## 6. Prepare the laptop

Create the account you'll sign in with. It will ask for a password of at least 8 characters. Sign-ups through the website are now closed.

```powershell
cd "D:\IVP PROJECT\19. SafeFall_ AI-Powered Fall Detection & Alert System with Python & Computer Vision"
.venv/Scripts/python.exe create_user.py yourname
```

Install the tunnel, then close and reopen PowerShell:

```powershell
winget install --id Cloudflare.cloudflared
```

## 7. Run it

In the project folder, start SafeFall:

```powershell
.venv/Scripts/python.exe fall_detection.py
```

In a second PowerShell window, start the tunnel:

```powershell
cloudflared tunnel --url http://localhost:5000
```

It prints an address ending in `trycloudflare.com`. Enter that in the app with your username and password, and allow notifications when asked.

This free address changes every time the tunnel restarts, so you'll need to sign in with the new one. Anyone with the address can reach the sign-in page, so use a strong password.

## 8. Test it

Turn Wi-Fi off on the phone and close the app. The camera clips loop, so a fall alert should arrive within a minute or two.

Check that:

- [ ] The notification arrives with sound while the app is closed
- [ ] Tapping it opens the alert with its photo
- [ ] Live view works on mobile data
- [ ] **I'm responding** stops the "still down" reminder for that alert
- [ ] Telegram still gets the alert

If the alert doesn't arrive, turn off battery optimisation for SafeFall in your phone's settings. Some Android brands delay notifications to save battery.

## Settings in `.env`

| Setting | Default | Meaning |
|---|---|---|
| `FALL_MODEL` | `off` | `on` uses the trained model: catches more falls, but raises more false alarms |
| `ALERT_COOLDOWN_SECONDS` | `120` | A repeat fall on a camera joins its open alert instead of alerting again |
| `ALLOW_REGISTRATION` | `0` | Keep at 0 while the tunnel is running |
