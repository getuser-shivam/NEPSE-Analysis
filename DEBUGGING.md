# NEPSE Analysis - Wireless Debugging Guide

This guide explains how to set up wireless debugging for the NEPSE Analysis web application.

## Wireless Debugging Setup

### 1. Local Development with Hot Reload

#### Flutter Web Development
```bash
# Navigate to mobile app directory
cd apps/mobile

# Run Flutter web with hot reload
flutter run -d chrome --web-port 8080
```

#### Backend API Development
```bash
# Navigate to API directory
cd apps/api

# Start API server with file watching
npm run dev
```

### 2. Network Access Configuration

#### Accessing Local Development Server from Other Devices

**Find your local IP address:**
- Windows: `ipconfig` (look for IPv4 Address)
- macOS/Linux: `ifconfig` or `ip addr show`

**Configure Flutter to accept network connections:**
```bash
flutter run -d chrome --web-hostname 0.0.0.0 --web-port 8080
```

**Access from other devices:**
```
http://YOUR_LOCAL_IP:8080
```

### 3. CORS Configuration for Development

The backend is already configured with CORS. Update `.env` for development:

```env
# Allow connections from any origin during development
CORS_ORIGIN=*
```

### 4. Tunneling Services for Remote Access

#### Using ngrok (Recommended)

1. Install ngrok: https://ngrok.com/download
2. Start ngrok tunnel:
```bash
ngrok http 8080
```
3. Use the provided HTTPS URL to access your local development server from anywhere

#### Using LocalTunnel

```bash
npx localtunnel --port 8080
```

### 5. Chrome DevTools Remote Debugging

#### Enable Remote Debugging in Flutter Web

1. Run Flutter web in debug mode:
```bash
flutter run -d chrome --debug
```

2. Open Chrome DevTools:
   - Press F12 or right-click → Inspect
   - Navigate to Sources tab for debugging
   - Use Network tab to monitor API calls

#### Flutter DevTools

```bash
# Start Flutter DevTools
flutter pub global activate devtools
flutter pub global run devtools
```

Access DevTools at: http://localhost:9100

### 6. Wireless Debugging for Mobile Devices

#### iOS (Safari)

1. Enable Web Inspector on iOS device:
   - Settings → Safari → Advanced → Web Inspector
   
2. Connect iOS device to Mac via USB
   
3. On Mac: Safari → Develop → [Your Device] → [Web Page]

#### Android (Chrome)

1. Enable USB Debugging on Android device:
   - Settings → Developer Options → USB Debugging
   
2. Connect Android device to computer via USB
   
3. On computer, open Chrome:
   - Navigate to `chrome://inspect`
   - Select your device and web page

### 7. WebSocket Configuration for Live Reload

The Flutter web app already supports hot reload. For custom WebSocket debugging:

**Backend WebSocket Configuration:**
```javascript
// In apps/api/src/server.js
import { createServer } from 'http';
import { WebSocketServer } from 'ws';

const wss = new WebSocketServer({ port: 8081 });

wss.on('connection', (ws) => {
  console.log('New WebSocket connection');
  
  ws.on('message', (message) => {
    console.log('Received:', message);
  });
});
```

**Frontend WebSocket Connection:**
```javascript
const ws = new WebSocket('ws://YOUR_IP:8081');

ws.onmessage = (event) => {
  console.log('Received:', event.data);
};
```

### 8. Performance Monitoring

#### Lighthouse CI

```bash
# Install Lighthouse CI
npm install -g @lhci/cli

# Run Lighthouse
lhci autorun
```

#### Flutter Performance Overlay

```bash
flutter run --profile --performance-overlay
```

### 9. Debugging Checklist

- [ ] Local development server running on accessible port
- [ ] CORS configured for network access
- [ ] Firewall allows inbound connections on development port
- [ ] Devices on same network can access local IP
- [ ] Chrome DevTools accessible for debugging
- [ ] WebSocket connections working for live reload
- [ ] Error logging enabled in console
- [ ] Network tab monitoring API calls

### 10. Troubleshooting

#### Connection Refused
- Check if server is running
- Verify firewall settings
- Ensure correct IP address and port

#### CORS Errors
- Verify CORS_ORIGIN in .env
- Check if backend is accepting requests from your origin
- Use browser extensions to bypass CORS for development only

#### Hot Reload Not Working
- Ensure running in debug mode (not release)
- Check console for WebSocket connection errors
- Restart development server

#### Mobile Device Cannot Connect
- Ensure devices on same WiFi network
- Check if device can ping development machine
- Verify firewall allows connections from device IP

## Production Debugging

For production debugging on GitHub Pages:

1. **Browser Console**: Always check browser console for errors
2. **Network Tab**: Monitor API calls and responses
3. **Performance**: Use Lighthouse to audit performance
4. **Error Tracking**: Consider integrating Sentry or similar service
5. **Analytics**: Use GitHub Pages analytics or Google Analytics

## Security Notes

- Never expose production credentials in wireless debugging
- Use separate development environment
- Disable remote debugging in production
- Use VPN for secure remote access when needed
- Rotate debug tokens regularly
