# Android Debugging Setup Guide

Complete guide for USB and wireless debugging of the NEPSE Analysis Flutter application.

## Prerequisites

- Android device with Android 5.0+ (API level 21+)
- USB cable
- Development machine with Windows/macOS/Linux
- Flutter SDK installed
- Android Studio or VS Code with Flutter extension

---

## Part 1: Enable Developer Options on Android Device

### Step 1: Enable Developer Options

1. Open **Settings** on your Android device
2. Scroll down and tap **About phone** (or **About device**)
3. Find **Build number** and tap it **7 times**
4. You'll see a toast message: "You are now a developer!"

### Step 2: Enable USB Debugging

1. Go back to **Settings** → **System** → **Developer options**
2. Find **USB debugging** and toggle it **ON**
3. Tap **OK** on the warning dialog

### Step 3: Enable Wireless Debugging (Android 11+)

1. In **Developer options**, find **Wireless debugging**
2. Toggle it **ON**
3. Tap **Allow** on the warning dialog
4. Note the **IP address and port** displayed (e.g., `192.168.1.100:42073`)

---

## Part 2: Install ADB (Android Debug Bridge)

### Option A: Using Android Studio (Recommended)

ADB is automatically installed with Android Studio:

```bash
# ADB location on Windows (default)
C:\Users\%USERNAME%\AppData\Local\Android\Sdk\platform-tools\adb.exe

# Add to PATH
setx PATH "%PATH%;C:\Users\%USERNAME%\AppData\Local\Android\Sdk\platform-tools"
```

### Option B: Using SDK Manager Command Line

```bash
# Install platform-tools via SDK manager
sdkmanager "platform-tools"
```

### Option C: Standalone ADB Installation

#### Windows

1. Download platform-tools from: https://developer.android.com/studio/releases/platform-tools
2. Extract to `C:\platform-tools`
3. Add to system PATH:
   ```powershell
   # PowerShell (as Administrator)
   [Environment]::SetEnvironmentVariable(
       "Path",
       $env:Path + ";C:\platform-tools",
       "User"
   )
   ```

#### macOS

```bash
# Using Homebrew
brew install android-platform-tools

# Or manually
curl -O https://dl.google.com/android/repository/platform-tools-latest-darwin.zip
unzip platform-tools-latest-darwin.zip
sudo mv platform-tools /usr/local/bin/
echo 'export PATH=$PATH:/usr/local/bin/platform-tools' >> ~/.zshrc
```

#### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install android-tools-adb android-tools-fastboot

# Or download manually
wget https://dl.google.com/android/repository/platform-tools-latest-linux.zip
unzip platform-tools-latest-linux.zip
sudo mv platform-tools /opt/
echo 'export PATH=$PATH:/opt/platform-tools' >> ~/.bashrc
```

### Verify ADB Installation

```bash
# Check ADB version
adb version

# Expected output:
# Android Debug Bridge version 1.0.41
# Version 34.0.0-xxxxxxx
```

---

## Part 3: USB Debugging Setup

### Step 1: Connect Device via USB

1. Connect your Android device to your computer using a USB cable
2. On your device, tap **Allow** when prompted with "Allow USB debugging?"
3. Check **Always allow from this computer** to avoid future prompts

### Step 2: Verify Device Connection

```bash
# List connected devices
adb devices

# Expected output:
# List of devices attached
# xxxxxxxx    device
```

### Step 3: Check Device Authorization

```bash
# Check device status
adb devices -l

# If device shows "unauthorized":
# 1. Disconnect USB
# 2. Revoke USB debugging authorizations (in Developer options)
# 3. Reconnect and accept prompt
```

---

## Part 4: Wireless Debugging Setup

### Method A: Using Android 11+ Wireless Debugging (Easiest)

#### Step 1: Pair Device (One-time)

1. On your Android device:
   - Go to **Settings** → **Developer options** → **Wireless debugging**
   - Tap **Pair with pairing code**
   - Note the **pairing code** and **IP:PORT**

2. On your development machine:
   ```bash
   # Pair using the code
   adb pair 192.168.1.100:42073
   # Enter pairing code: 123456
   ```

#### Step 2: Connect Wirelessly

```bash
# Connect to device using the displayed IP:PORT
adb connect 192.168.1.100:42073

# Verify connection
adb devices

# Expected output:
# List of devices attached
# 192.168.1.100:42073    device
```

### Method B: Using USB + TCP/IP (Works on all Android versions)

#### Step 1: Connect via USB First

```bash
# Ensure device is connected via USB
adb devices

# Output should show:
# xxxxxxxx    device
```

#### Step 2: Enable TCP/IP Mode

```bash
# Switch ADB to TCP/IP mode on port 5555
adb tcpip 5555

# Expected output:
# restarting in TCP mode port: 5555
```

#### Step 3: Connect Wirelessly

```bash
# Find your device's IP address
# Method 1: In Developer options → Wireless debugging
# Method 2: Settings → About phone → Status → IP address
# Method 3: Using ADB
adb shell ip route | awk '{print $9}'

# Connect wirelessly (replace with your device's IP)
adb connect 192.168.1.100:5555

# Verify connection
adb devices

# Expected output:
# List of devices attached
# 192.168.1.100:5555    device
```

#### Step 4: Disconnect USB (Optional)

Once wireless connection is established, you can disconnect the USB cable.

---

## Part 5: Flutter Debug Configuration

### VS Code Setup

#### Step 1: Install Extensions

1. Install **Flutter** extension
2. Install **Dart** extension
3. Install **ADB Interface for VSCode** (optional, for ADB commands)

#### Step 2: Configure launch.json

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "NEPSE Analysis (USB)",
            "request": "launch",
            "type": "dart",
            "program": "nepse_app/lib/main.dart",
            "args": [
                "--device-id",
                "<YOUR_DEVICE_ID>"
            ]
        },
        {
            "name": "NEPSE Analysis (Wireless)",
            "request": "launch",
            "type": "dart",
            "program": "nepse_app/lib/main.dart",
            "args": [
                "--device-id",
                "<WIRELESS_IP:PORT>"
            ]
        },
        {
            "name": "NEPSE Analysis (Profile Mode)",
            "request": "launch",
            "type": "dart",
            "program": "nepse_app/lib/main.dart",
            "flutterMode": "profile",
            "args": [
                "--device-id",
                "<DEVICE_ID>"
            ]
        }
    ]
}
```

Replace `<YOUR_DEVICE_ID>` with the ID from `adb devices`.

#### Step 3: Debug Configuration

Add `.vscode/settings.json`:

```json
{
    "dart.flutterSdkPath": "C:\\flutter\\flutter",
    "dart.openDevTools": "flutter",
    "dart.previewFlutterUiGuides": true,
    "dart.previewFlutterUiGuidesCustomTracking": true,
    "debug.allowBreakpointsEverywhere": true,
    "debug.inlineValues": true
}
```

### Android Studio Setup

#### Step 1: Configure Device Connection

1. Open **Android Studio**
2. Go to **Tools** → **SDK Manager**
3. Ensure **Android SDK Platform-Tools** is installed
4. Click **Apply** → **OK**

#### Step 2: Run Configuration

1. Open **Run** → **Edit Configurations**
2. Click **+** → **Flutter**
3. Set:
   - **Name**: NEPSE Analysis
   - **Dart entrypoint**: `nepse_app/lib/main.dart`
   - **Build flavor**: (leave empty)
   - **Target device**: Select your device

#### Step 3: Enable Wireless Debugging in Android Studio

1. Open **Device Manager** (View → Tool Windows → Device Manager)
2. Click **+** → **Pair using Wi-Fi**
3. Scan QR code or enter pairing code
4. Select device from list

---

## Part 6: Running and Debugging

### Start Debugging Session

#### Using VS Code

```bash
# Press F5 or run from Debug panel
# Or use command palette (Ctrl+Shift+P):
> Flutter: Run Flutter App
```

#### Using Command Line

```bash
# Navigate to project
cd nepse_app

# Run on specific device
flutter run --device-id <DEVICE_ID>

# Run with verbose logging
flutter run --verbose --device-id <DEVICE_ID>

# Run in profile mode (for performance testing)
flutter run --profile --device-id <DEVICE_ID>

# Run in release mode
flutter run --release --device-id <DEVICE_ID>
```

### Debugging Features

#### Set Breakpoints

1. Click on the left margin in VS Code/Android Studio
2. Red dot indicates active breakpoint
3. Use **F9** to toggle breakpoints

#### Inspect Variables

- Hover over variables to see values
- Use **Debug Console** (Ctrl+Shift+Y in VS Code)
- Add variables to **Watch** window

#### Step Through Code

- **F5**: Continue execution
- **F10**: Step over
- **F11**: Step into
- **Shift+F11**: Step out
- **Shift+F5**: Restart
- **Shift+F5**: Stop

#### Hot Reload / Hot Restart

- **r**: Hot reload (keeps state)
- **R**: Hot restart (resets state)
- Press in terminal where Flutter is running

---

## Part 7: Troubleshooting

### Device Not Found

```bash
# Check ADB server is running
adb start-server

# Restart ADB server
adb kill-server
adb start-server

# Check USB connection
adb devices -l
```

### Unauthorized Device

```bash
# 1. Revoke authorizations on device
#    Settings → Developer options → Revoke USB debugging authorizations

# 2. Disconnect and reconnect USB
# 3. Accept prompt on device

# 3. Restart ADB
adb kill-server
adb start-server
```

### Wireless Connection Fails

```bash
# Check firewall settings
telnet <DEVICE_IP> 5555

# Ensure same network
adb shell ip route

# Restart wireless debugging on device
# Then reconnect via USB and repeat setup

# Reset ADB to USB mode if needed
adb usb
```

### Flutter Device Not Detected

```bash
# List Flutter devices
flutter devices

# If empty, ensure ADB works
adb devices

# Check Flutter doctor
flutter doctor -v

# Accept Android licenses if needed
flutter doctor --android-licenses
```

### Port Already in Use

```bash
# Find process using port 5555
netstat -ano | findstr :5555

# Kill process
# Windows:
taskkill /PID <PID> /F

# macOS/Linux:
kill -9 <PID>
```

---

## Part 8: Advanced Configuration

### Custom Debugging Port

```bash
# Use different port for wireless debugging
adb tcpip 5556
adb connect 192.168.1.100:5556

# Multiple devices on different ports
adb tcpip 5555  # Device 1
adb tcpip 5556  # Device 2
```

### Network Interface Configuration

#### Windows (Using Specific Network Adapter)

```powershell
# Find network interfaces
Get-NetAdapter | Where-Object {$_.Status -eq "Up"}

# Use specific interface (advanced)
# Requires route configuration
```

#### macOS/Linux

```bash
# Bind ADB to specific interface
adb -a -P 5037 fork-server server

# Connect using specific interface
adb connect 192.168.1.100:5555
```

### Secure Connection (SSH Tunnel)

```bash
# Create SSH tunnel for secure remote debugging
ssh -L 5555:localhost:5555 user@remote-machine

# Connect via tunnel
adb connect localhost:5555
```

---

## Part 9: Scripts and Automation

### Quick Setup Script (Windows PowerShell)

Create `scripts/setup-debugging.ps1`:

```powershell
# ADB Wireless Debugging Setup Script

Write-Host "=== NEPSE Analysis Debugging Setup ===" -ForegroundColor Cyan

# Check if ADB is available
$adb = Get-Command adb -ErrorAction SilentlyContinue
if (-not $adb) {
    Write-Host "❌ ADB not found. Please install Android SDK platform-tools." -ForegroundColor Red
    exit 1
}

Write-Host "✅ ADB found: $(adb version)" -ForegroundColor Green

# List devices
Write-Host "`nConnected devices:" -ForegroundColor Yellow
adb devices -l

# Get device IP
Write-Host "`nDevice IP addresses:" -ForegroundColor Yellow
$ips = adb shell ip route | ForEach-Object { ($_ -split '\s+')[8] } | Sort-Object -Unique
$ips | ForEach-Object { Write-Host "  - $_" }

# Setup wireless
if ($ips) {
    $deviceIp = $ips[0]
    Write-Host "`nSetting up wireless debugging on $deviceIp..." -ForegroundColor Cyan
    
    # Enable TCP/IP
    adb tcpip 5555
    
    # Connect wirelessly
    Start-Sleep -Seconds 2
    adb connect "${deviceIp}:5555"
    
    # Verify
    Write-Host "`nVerifying connection..." -ForegroundColor Yellow
    adb devices
    
    Write-Host "`n✅ Wireless debugging setup complete!" -ForegroundColor Green
    Write-Host "You can now disconnect the USB cable." -ForegroundColor Cyan
} else {
    Write-Host "❌ Could not determine device IP. Ensure device is connected via USB." -ForegroundColor Red
}
```

### Quick Setup Script (Bash)

Create `scripts/setup-debugging.sh`:

```bash
#!/bin/bash

# NEPSE Analysis Debugging Setup Script

set -e

echo "=== NEPSE Analysis Debugging Setup ==="

# Check if ADB is available
if ! command -v adb &> /dev/null; then
    echo "❌ ADB not found. Please install Android SDK platform-tools."
    exit 1
fi

echo "✅ ADB found: $(adb version | head -1)"

# List devices
echo ""
echo "Connected devices:"
adb devices -l

# Get device IP
echo ""
echo "Device IP addresses:"
DEVICE_IP=$(adb shell ip route | awk '{print $9}' | head -1)

if [ -n "$DEVICE_IP" ]; then
    echo "  - $DEVICE_IP"
    
    echo ""
    echo "Setting up wireless debugging on $DEVICE_IP..."
    
    # Enable TCP/IP
    adb tcpip 5555
    
    # Wait for device to restart networking
    sleep 2
    
    # Connect wirelessly
    adb connect "${DEVICE_IP}:5555"
    
    # Verify
    echo ""
    echo "Verifying connection..."
    adb devices
    
    echo ""
    echo "✅ Wireless debugging setup complete!"
    echo "You can now disconnect the USB cable."
else
    echo "❌ Could not determine device IP. Ensure device is connected via USB."
    exit 1
fi
```

### Run Configuration Script

Create `scripts/run-debug.sh`:

```bash
#!/bin/bash

# Run NEPSE Analysis with debugging

# Find connected device
DEVICE=$(adb devices | grep -v "List" | grep "device" | head -1 | awk '{print $1}')

if [ -z "$DEVICE" ]; then
    echo "❌ No device found. Please connect a device."
    exit 1
fi

echo "Using device: $DEVICE"

# Navigate to project
cd "$(dirname "$0")/../nepse_app"

# Run Flutter app with debugging
echo "Starting Flutter app in debug mode..."
flutter run \
    --device-id "$DEVICE" \
    --debug \
    --verbose \
    --hot
```

---

## Part 10: VS Code Tasks

Add `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Setup Wireless Debugging",
            "type": "shell",
            "command": "${workspaceFolder}/scripts/setup-debugging.sh",
            "windows": {
                "command": "${workspaceFolder}/scripts/setup-debugging.ps1"
            },
            "problemMatcher": [],
            "presentation": {
                "reveal": "always",
                "panel": "new"
            }
        },
        {
            "label": "Connect Device",
            "type": "shell",
            "command": "adb connect ${input:deviceIp}:5555",
            "problemMatcher": []
        },
        {
            "label": "List Devices",
            "type": "shell",
            "command": "adb devices -l",
            "problemMatcher": []
        },
        {
            "label": "Flutter: Run (Debug)",
            "type": "shell",
            "command": "flutter run --debug",
            "options": {
                "cwd": "${workspaceFolder}/nepse_app"
            },
            "problemMatcher": [],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ],
    "inputs": [
        {
            "id": "deviceIp",
            "description": "Device IP Address",
            "default": "192.168.1.100",
            "type": "promptString"
        }
    ]
}
```

---

## Quick Reference Commands

### ADB Commands

```bash
# Device management
adb devices              # List devices
adb connect IP:PORT     # Connect wirelessly
adb disconnect IP:PORT  # Disconnect wirelessly
adb kill-server         # Stop ADB server
adb start-server        # Start ADB server
adb usb                 # Switch to USB mode
adb tcpip 5555          # Enable TCP/IP mode

# Shell access
adb shell               # Open shell on device
adb shell pm list packages  # List installed apps
adb shell screencap -p /sdcard/screenshot.png  # Screenshot
adb shell screenrecord /sdcard/recording.mp4  # Screen recording

# File transfer
adb push local.txt /sdcard/remote.txt    # Push file
adb pull /sdcard/remote.txt local.txt   # Pull file

# Install APK
adb install app.apk                    # Install app
adb install -r app.apk                 # Reinstall/upgrade
adb uninstall com.example.app          # Uninstall app

# Logs
adb logcat                             # View logs
adb logcat -d > logcat.txt             # Save logs to file
adb logcat -s Flutter                  # Filter Flutter logs
adb logcat *:E                         # Show errors only

# Debug
adb shell am start -D -n com.example.app/.MainActivity  # Debug mode
adb shell dumpsys activity | grep -i run              # Running apps
```

### Flutter Commands

```bash
# Device management
flutter devices                        # List devices
flutter emulators                      # List emulators

# Run app
flutter run                            # Run on default device
flutter run -d <DEVICE_ID>            # Run on specific device
flutter run --debug                    # Debug mode
flutter run --profile                # Profile mode
flutter run --release                # Release mode
flutter run --verbose                # Verbose logging

# Build
flutter build apk                      # Build APK
flutter build apk --release          # Release APK
flutter build appbundle              # Build App Bundle

# Debug
flutter attach                         # Attach to running app
flutter screenshot                     # Take screenshot

# Logs
flutter logs                           # View logs
flutter logs --clear                 # Clear and view logs

# Clean
flutter clean                          # Clean build files
```

---

## Security Best Practices

1. **Disable USB debugging when not needed**
   - Settings → Developer options → USB debugging → OFF

2. **Use wireless debugging on trusted networks only**
   - Avoid public Wi-Fi for debugging
   - Use VPN if debugging remotely

3. **Revoke authorizations regularly**
   - Settings → Developer options → Revoke USB debugging authorizations

4. **Monitor connected devices**
   ```bash
   adb devices
   # Disconnect unknown devices immediately
   adb disconnect <UNKNOWN_IP>
   ```

5. **Use app-specific debugging ports**
   - Don't use default port 5555 in production
   - Use randomized high ports (e.g., 49000-65000)

---

## Support

For debugging issues:
1. Check `flutter doctor -v`
2. Review `adb logcat` output
3. Check Android Studio/Device Manager
4. Verify network connectivity: `ping <DEVICE_IP>`

For Flutter-specific issues:
- [Flutter Debugging Docs](https://flutter.dev/docs/development/tools/devtools)
- [Android Studio Setup](https://developer.android.com/studio/intro)
