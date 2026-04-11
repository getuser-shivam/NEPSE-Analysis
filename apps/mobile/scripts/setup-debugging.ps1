# NEPSE Analysis Debugging Setup Script (Windows PowerShell)
# This script sets up wireless debugging for the NEPSE Analysis Flutter app

Write-Host "=== NEPSE Analysis Debugging Setup ===" -ForegroundColor Cyan
Write-Host "This script will configure wireless debugging for your Android device" -ForegroundColor Gray

# Check if ADB is available
$adb = Get-Command adb -ErrorAction SilentlyContinue
if (-not $adb) {
    Write-Host "`n❌ ADB not found in PATH!" -ForegroundColor Red
    Write-Host "Please install Android SDK platform-tools:" -ForegroundColor Yellow
    Write-Host "  1. Install Android Studio: https://developer.android.com/studio" -ForegroundColor White
    Write-Host "  2. Or download standalone: https://developer.android.com/studio/releases/platform-tools" -ForegroundColor White
    Write-Host "`nThen add to your PATH:" -ForegroundColor Yellow
    Write-Host "  C:\Users\%USERNAME%\AppData\Local\Android\Sdk\platform-tools" -ForegroundColor White
    exit 1
}

Write-Host "`n✅ ADB found: $(adb version | Select-Object -First 1)" -ForegroundColor Green

# Check for connected USB devices
Write-Host "`nChecking for connected devices..." -ForegroundColor Yellow
$devices = adb devices | Select-String "device$" | ForEach-Object { $_.Line }

if (-not $devices) {
    Write-Host "❌ No USB devices found!" -ForegroundColor Red
    Write-Host "`nPlease:" -ForegroundColor Yellow
    Write-Host "  1. Connect your Android device via USB" -ForegroundColor White
    Write-Host "  2. Enable USB Debugging in Developer Options" -ForegroundColor White
    Write-Host "  3. Accept the USB debugging prompt on your device" -ForegroundColor White
    exit 1
}

Write-Host "✅ Device connected via USB:" -ForegroundColor Green
$devices | ForEach-Object { Write-Host "   $_" -ForegroundColor Gray }

# Get device IP addresses
Write-Host "`nGetting device IP addresses..." -ForegroundColor Yellow
try {
    $ipOutput = adb shell ip route 2>$null
    $ips = $ipOutput | ForEach-Object {
        $parts = $_ -split '\s+'
        if ($parts.Count -ge 9) { $parts[8] }
    } | Where-Object { $_ -and ($_ -ne '0.0.0.0') } | Sort-Object -Unique

    if (-not $ips) {
        # Alternative method
        $ipOutput = adb shell ifconfig 2>$null | Select-String "inet\s+([0-9.]+)"
        $ips = $ipOutput | ForEach-Object {
            if ($_ -match "inet\s+([0-9.]+)") { $matches[1] }
        } | Where-Object { $_ -and ($_ -ne '127.0.0.1') }
    }

    if (-not $ips) {
        Write-Host "❌ Could not determine device IP address!" -ForegroundColor Red
        Write-Host "Please manually find your device's IP in Settings > About Phone > Status" -ForegroundColor Yellow
        exit 1
    }

    Write-Host "✅ Found IP addresses:" -ForegroundColor Green
    $ips | ForEach-Object { Write-Host "   - $_" -ForegroundColor Gray }
} catch {
    Write-Host "❌ Error getting IP address: $_" -ForegroundColor Red
    exit 1
}

# Use first IP
$deviceIp = $ips[0]
Write-Host "`nUsing IP: $deviceIp" -ForegroundColor Cyan

# Enable TCP/IP mode
Write-Host "`nEnabling TCP/IP mode (port 5555)..." -ForegroundColor Yellow
try {
    $tcpipOutput = adb tcpip 5555 2>&1
    Write-Host "✅ $tcpipOutput" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to enable TCP/IP mode: $_" -ForegroundColor Red
    exit 1
}

# Wait for device to restart networking
Write-Host "Waiting for device to restart networking..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Connect wirelessly
Write-Host "`nConnecting wirelessly to ${deviceIp}:5555..." -ForegroundColor Yellow
try {
    $connectOutput = adb connect "${deviceIp}:5555" 2>&1
    if ($connectOutput -match "connected|already connected") {
        Write-Host "✅ $connectOutput" -ForegroundColor Green
    } else {
        Write-Host "⚠️ $connectOutput" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Failed to connect: $_" -ForegroundColor Red
    exit 1
}

# Verify connection
Write-Host "`nVerifying wireless connection..." -ForegroundColor Yellow
Start-Sleep -Seconds 2
try {
    $deviceList = adb devices 2>$null
    $wirelessDevice = $deviceList | Select-String "${deviceIp}:5555"
    
    if ($wirelessDevice) {
        Write-Host "✅ Wireless connection established!" -ForegroundColor Green
        Write-Host "   $wirelessDevice" -ForegroundColor Gray
    } else {
        Write-Host "⚠️ Connection may not be fully established yet" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️ Could not verify connection: $_" -ForegroundColor Yellow
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   ✅ Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Device IP: $deviceIp" -ForegroundColor White
Write-Host "Port: 5555" -ForegroundColor White
Write-Host "`nYou can now:" -ForegroundColor Yellow
Write-Host "  1. Disconnect the USB cable" -ForegroundColor White
Write-Host "  2. Run: flutter run --device-id ${deviceIp}:5555" -ForegroundColor White
Write-Host "  3. Debug wirelessly!" -ForegroundColor White
Write-Host "`nTo disconnect:" -ForegroundColor Yellow
Write-Host "  adb disconnect ${deviceIp}:5555" -ForegroundColor Gray
Write-Host "`n" -ForegroundColor Reset
