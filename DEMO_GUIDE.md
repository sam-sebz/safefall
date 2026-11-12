# SafeFall review: demo guide

## Tonight
- [ ] **Record a backup video:** a fall clip on the laptop, then the phone notification, opening the alert, and tapping "I'm responding". Use the phone's screen recorder. If the internet fails tomorrow, play this instead.
- [ ] **Charge both devices,** and bring the laptop charger. On battery the laptop slows down.
- [ ] **Practise the whole run once** (below) and time it.
- [ ] **Check the phone:** SafeFall is installed, notifications are allowed, and battery optimisation is off for SafeFall.
- [ ] **Know your password** for the `sam` account.

## At the venue, 10 minutes before

Open two PowerShell windows in the project folder:

```powershell
cd "D:\IVP PROJECT\19. SafeFall_ AI-Powered Fall Detection & Alert System with Python & Computer Vision"
```

**Window 1, start SafeFall:**

```powershell
.venv/Scripts/python.exe fall_detection.py
```

**Window 2, start the tunnel:**

```powershell
& "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:5000
```

1. Copy the new `https://....trycloudflare.com` address from window 2. It changes every time.
2. In the phone app, sign out (bottom of **Cameras**), then sign in with the new address.
3. Wait for one alert to arrive, so you know everything works before you present.
4. On the laptop, open `http://localhost:5000` in a browser and sign in, so the web dashboard is ready.

If the venue Wi-Fi blocks the tunnel, use your phone's hotspot for the laptop.

## Running order (about 10 minutes)

| Time | Show | Say |
|---|---|---|
| 1 min | Title slide | Falls in factories and warehouses can go unseen. SafeFall watches CCTV and alerts a manager's phone with the location. |
| 1 min | How it works | Camera → YOLO11 pose and tracking → fall decision → saved alert → phone push and Telegram |
| 3 min | **Live demo:** laptop dashboard | Six cameras, each with a building and location. Point out the boxes and the body keypoints. |
| | **Live demo:** phone | Open **Cameras** and the live view. Then wait for a fall: the notification shows the location. |
| | **Live demo:** respond | Tap the notification to show the photo and time, tap **I'm responding**, and show it's saved under your name. Show the Telegram backup message. |
| 2 min | Results | The table below |
| 1 min | Limitations, future work | The list below |
| | Questions | |

**Tip:** if an alert is slow to come, open **Alerts** and show an earlier one, then explain that a new alert waits 2 minutes per camera so the looping demo clips don't flood the phone.

## Results (UR Fall dataset, clips held back from tuning)

| Method | Falls caught | False alarms |
|---|---|---|
| Rules, full analysis | 15 of 15 | 1 of 20 |
| Rules, 7.5 frames a second | 11 of 15 | 1 of 20 |
| Trained model, 7.5 frames a second | 14 of 15 | 6 of 20 |

**How to explain it:** settings were tuned on odd-numbered clips and scored once on even-numbered clips they never saw. The trained model catches more falls but gives too many false alarms, so the rules stay the default. Collecting more training data is the next step.

**If asked about live speed:** these scores assume 7.5 frames a second. On this laptop with six cameras, each one actually gets about 3 to 4 frames a second, so live accuracy may be lower than the table shows.

## Limitations (be honest about these)
- Tested on staged clips in rooms, not real factory CCTV.
- The demo cameras are video files, not live IP cameras.
- With six cameras on a laptop CPU, each camera gets only about 3 to 4 frames a second. Detection paused twice during one test on battery, but a 5-minute test on the charger had no pauses, and the cause isn't confirmed.
- Tracking sometimes gives one person two IDs, so the alert can say "2 down" when only one person fell. The Workshop camera did this once during testing.
- Each person is seen by one camera only, so a person hidden behind something can be missed.
- Alert photos show identifiable people.

## Likely questions
- **Why pose instead of a bounding box?** Torso angle tells lying from crouching better than box shape. Box shape is the fallback when keypoints aren't visible.
- **How does it know the location?** Each camera has a location set in `cameras.json`, and the alert uses it.
- **What about several people?** Each person is tracked with their own ID, and the alert says how many are down.
- **Why not alert on any lying person?** A fall needs a sudden drop. Someone found lying with no fall seen waits longer before alerting, to avoid false alarms.
- **What if nobody responds?** A "still down" reminder is sent, and it stops once someone taps "I'm responding".
- **Is it secure?** Sign-up is closed, accounts are added by an admin, every API call needs a sign-in token, and repeated wrong passwords are blocked.
- **How do alerts reach the phone when the app is closed?** Expo's push service over Firebase Cloud Messaging.
- **Why is it not production-ready?** Flask development server, a temporary tunnel address, and no real-site testing.

Everyone in the team should be ready to explain any part of the code and results, because viva questions often go into detail.

## If something goes wrong
| Problem | Fix |
|---|---|
| Phone says it can't reach the server | Check both windows are running, and sign in again with the current tunnel address |
| No notification | Open **Alerts** in the app, or show Telegram, and explain push delivery can be delayed |
| Laptop very slow | Plug it in. As a last resort, play the backup video |
| Tunnel won't start | Switch the laptop to your phone's hotspot and try again |
