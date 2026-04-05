# Dart + Node + Prisma Stack

As of 2026-04-05, this repository does not contain an existing SQL connection string or database credential to reuse. The current Python application persists state in local pickle files such as `portfolio.pkl`, `watchlist.pkl`, and `nepse_data.pkl`.

This starter stack adds a parallel implementation path without removing the existing Python app:

- `backend/`: Node.js API with Express and Prisma
- `backend/prisma/schema.prisma`: relational schema for settings, portfolio holdings, watchlist items, and price snapshots
- `backend/sql/init.sql`: SQL reference script for the same data model
- `dart_client/`: Dart package that calls the new API

The Dart side now also follows a more app-oriented shape inspired by `I:\Path\Projects\iFutures`:

- `models/` for DTOs
- `services/` for higher-level snapshot loading
- a dashboard snapshot endpoint and client flow so a UI can request one aggregated payload instead of assembling each card separately

## Backend Setup

From `backend/`:

```bash
copy .env.example .env
npm install
npm run prisma:generate
npm run prisma:push
npm run seed
npm run dev
```

The default local development database uses SQLite:

```env
DATABASE_URL="file:./dev.db"
```

## Dart Client Setup

From `dart_client/`:

```bash
dart pub get
dart run example/main.dart
```

The example expects the API to be running at `http://localhost:4000`.

## API Endpoints

- `GET /health`
- `GET /api/dashboard`
- `GET /api/settings`
- `PATCH /api/settings`
- `GET /api/portfolio`
- `POST /api/portfolio`
- `PATCH /api/portfolio/:symbol`
- `DELETE /api/portfolio/:symbol`
- `GET /api/watchlist`
- `POST /api/watchlist`
- `DELETE /api/watchlist/:symbol`
- `GET /api/prices/:symbol`
- `POST /api/prices`

## Moving to PostgreSQL or MySQL

If you want a server database instead of SQLite, update two things:

1. Change `provider` in `backend/prisma/schema.prisma`
2. Replace `DATABASE_URL` in `backend/.env`

Example PostgreSQL URL:

```env
DATABASE_URL="postgresql://username:password@localhost:5432/nepse_analysis?schema=public"
```

No existing credential was found in this repository, so the connection values need to come from your target SQL server.
