#!/bin/bash

# Run NEPSE Analysis with wireless debugging
# Usage: ./run-debug.sh [device_ip]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}=== NEPSE Analysis Debug Runner ===${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if device IP is provided
DEVICE_IP="${1:-}"

if [ -z "$DEVICE_IP" ]; then
    # Try to find wireless device
    echo -e "${YELLOW}No device IP provided, searching for wireless devices...${NC}"
    DEVICE_IP=$(adb devices | grep -E '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+' | head -1 | awk '{print $1}')
    
    if [ -z "$DEVICE_IP" ]; then
        echo -e "${RED}❌ No wireless device found!${NC}"
        echo -e "${YELLOW}Please:${NC}"
        echo -e "  1. Run setup-debugging.sh first, or${NC}"
        echo -e "  2. Provide device IP: ./run-debug.sh 192.168.1.100${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Found wireless device: $DEVICE_IP${NC}"
fi

# Verify device is connected
echo -e "${YELLOW}Verifying device connection...${NC}"
if ! adb devices | grep -q "$DEVICE_IP"; then
    echo -e "${RED}❌ Device $DEVICE_IP not found!${NC}"
    echo -e "${YELLOW}Trying to connect...${NC}"
    adb connect "$DEVICE_IP"
fi

# Navigate to project
echo -e "${YELLOW}Navigating to project directory...${NC}"
cd "$PROJECT_DIR"

# Check Flutter
if ! command -v flutter &> /dev/null; then
    echo -e "${RED}❌ Flutter not found!${NC}"
    echo -e "${YELLOW}Please install Flutter: https://flutter.dev/docs/get-started/install${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Flutter found:${NC} $(flutter --version | head -1)"

# Get dependencies
echo -e "${YELLOW}Getting dependencies...${NC}"
flutter pub get

# Run Flutter app in debug mode
echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${GREEN}   Starting Flutter App${NC}"
echo -e "${CYAN}========================================${NC}"
echo -e "${WHITE}Device:${NC} $DEVICE_IP"
echo -e "${WHITE}Mode:${NC} Debug"
echo -e "${WHITE}Hot Reload:${NC} Enabled (press 'r' to reload)"
echo ""

flutter run \
    --device-id "$DEVICE_IP" \
    --debug \
    --hot \
    --verbose
