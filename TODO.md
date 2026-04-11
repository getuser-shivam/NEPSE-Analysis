# NEPSE Analysis - Project TODO List

## 🚀 High Priority (P0) - Core & Documentation
- [ ] **Sync Documentation**: Update `README.md` with the new multi-stack architecture (Backend, Flutter App, Dart Client).
- [ ] **Update File Structure**: Correct the file structure diagram in `README.md` (move core tools to `tools/`).
- [ ] **Technical Reference**: Ensure all new API endpoints in `backend/` are documented.

## 📱 Mobile Development (nepse_app) - P1
- [ ] **Watchlist Screen**: Implement a dedicated screen for managing and viewing the watchlist.
- [ ] **Portfolio Details**: Add detailed view for individual portfolio holdings with performance history.
- [ ] **Technical Charts**: Integrate interactive technical charts (similar to desktop) into the mobile app.
- [ ] **Real-time Price Sync**: Implement background fetching for price alerts.

## 🌐 Web Development (dart_client) - P1
- [ ] **Dashboard Implementation**: Replace the "Loading..." placeholder with a functional dashboard.
- [ ] **Web Indicators**: Port technical indicators to the web interface.
- [ ] **User Auth**: Connect the web client to the `backend/` auth system.

## ⚙️ Backend Enhancements - P2
- [ ] **PostgreSQL Migration**: Support PostgreSQL/MySQL as alternatives to SQLite for production.
- [ ] **ML Prediction Service**: Implement the planned machine learning price prediction endpoint.
- [ ] **Real-time Streaming**: Integrate WebSockets for real-time price updates.

## 🎨 UI/UX & Polish - P3
- [ ] **Glassmorphic UI**: Apply glassmorphism styling to both Web and Mobile apps for a premium feel.
- [ ] **Performance Benchmarking**: Regular performance tests to ensure 100% test coverage and memory optimization.
- [ ] **Theme Refinement**: Harmonize light/dark themes across all three platforms (Desktop, Web, Mobile).

---
*Last updated: 2026-04-11*
