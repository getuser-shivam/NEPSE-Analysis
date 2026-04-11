#!/bin/bash

# NEPSE Analysis Debugging Setup Script (macOS/Linux)
# This script sets up wireless debugging for the NEPSE Analysis Flutter app

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

echo -e "${CYAN}=== NEPSE Analysis Debugging Setup ===${NC}"
echo -e "${GRAY}This script will configure wireless debugging for your Android device${NC}"

# Check if ADB is available
if ! command -v adb &> /dev/null; then
    echo -e "${RED}❌ ADB not found in PATH!${NC}"
    echo -e "${YELLOW}Please install Android SDK platform-tools:${NC}"
    echo -e "  ${WHITE}1. Install Android Studio: https://developer.android.com/studio${NC}"
    echo -e "  ${WHITE}2. Or download standalone: https://developer.android.com/studio/releases/platform-tools${NC}"
    echo -e "${YELLOW}Then add to your PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✅ ADB found:${NC} $(adb version | head -1)"

# Check for connected USB devices
echo -e "${YELLOW}Checking for connected devices...${NC}"
DEVICES=$(adb devices | grep -E "^[a-zA-Z0-9].*device$" || true)

if [ -z "$DEVICES" ]; then
    echo -e "${RED}❌ No USB devices found!${NC}"
    echo -e "${YELLOW}Please:${NC}"
    echo -e "  ${WHITE}1. Connect your Android device via USB${NC}"
    echo -e "  ${WHITE}2. Enable USB Debugging in Developer Options${NC}"
    echo -e "  ${WHITE}3. Accept the USB debugging prompt on your device${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Device connected via USB:${NC}"
echo "$DEVICES" | while read -r line; do
    echo -e "   ${GRAY}$line${NC}"
done

# Get device IP addresses
echo -e "${YELLOW}Getting device IP addresses...${NC}"
DEVICE_IP=""

# Try ip route first
if adb shell ip route &> /dev/null; then
    DEVICE_IP=$(adb shell ip route 2>/dev/null | awk '{print $9}' | grep -v '^$' | head -1)
fi

# Fallback to ifconfig
if [ -z "$DEVICE_IP" ]; then
    DEVICE_IP=$(adb shell ifconfig 2>/dev/null | grep -oE 'inet [0-9.]+' | grep -v '127.0.0.1' | head -1 | awk '{print $2}')
fi

# Alternative: use ip addr
if [ -z "$DEVICE_IP" ]; then
    DEVICE_IP=$(adb shell ip addr show wlan0 2>/dev/null | grep -oE 'inet [0-9.]+' | head -1 | awk '{print $2}')
fi

if [ -z "$DEVICE_IP" ]; then
    echo -e "${RED}❌ Could not determine device IP address!${NC}"
    echo -e "${YELLOW}Please manually find your device's IP in Settings > About Phone > Status${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found IP address:${NC}"
echo -e "   ${GRAY}$DEVICE_IP${NC}"

# Enable TCP/IP mode
echo -e "${YELLOW}Enabling TCP/IP mode (port 5555)...${NC}"
if adb tcpip 5555; then
    echo -e "${GREEN}✅ TCP/IP mode enabled${NC}"
else
    echo -e "${RED}❌ Failed to enable TCP/IP mode${NC}"
    exit 1
fi

# Wait for device to restart networking
echo -e "${YELLOW}Waiting for device to restart networking...${NC}"
sleep 3

# Connect wirelessly
echo -e "${YELLOW}Connecting wirelessly to $DEVICE_IP:5555...${NC}"
if adb connect "${DEVICE_IP}:5555"; then
    echo -e "${GREEN}✅ Connected wirelessly${NC}"
else
    echo -e "${RED}❌ Failed to connect wirelessly${NC}"
    exit 1
fi

# Verify connection
echo -e "${YELLOW}Verifying wireless connection...${NC}"
sleep 2
DEVICE_LIST=$(adb devices 2>/dev/null || true)
if echo "$DEVICE_LIST" | grep -q "${DEVICE_IP}:5555"; then
    echo -e "${GREEN}✅ Wireless connection verified!${NC}"
else
    echo -e "${YELLOW}⚠️ Connection may not be fully established yet${NC}"
fi

# Summary
echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${GREEN}   ✅ Setup Complete!${NC}"
echo -e "${CYAN}========================================${NC}"
echo -e "${WHITE}Device IP:${NC} $DEVICE_IP"
echo -e "${WHITE}Port:${NC} 5555"
echo ""
echo -e "${YELLOW}You can now:${NC}"
echo -e "  ${WHITE}1. Disconnect the USB cable${NC}"
echo -e "  ${WHITE}2. Run: flutter run --device-id ${DEVICE_IP}:5555${NC}"
echo -e "  ${WHITE}3. Debug wirelessly!${NC}"
echo ""
echo -e "${YELLOW}To disconnect:${NC}"
echo -e "  ${GRAY}adb disconnect ${DEVICE_IP}:5555${NC}"
echo ""
