# NEPSE Analysis Tool

A comprehensive stock analysis application for the Nepal Stock Exchange (NEPSE) with advanced portfolio management, technical indicators, and enterprise-grade features.

## 🚀 Features

### Core Functionality
- **Real-time Stock Data**: Fetch live stock data from NEPSE and Yahoo Finance APIs
- **Advanced Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R
- **Portfolio Management**: Track investments, calculate gains/losses, and performance metrics
- **Watchlist System**: Monitor multiple stocks simultaneously
- **Interactive Charts**: Zoom, pan, and save charts with multiple indicators

### Advanced Features
- **Portfolio Analytics**: Sharpe ratio, beta calculation, sector concentration analysis
- **Data Import/Export**: CSV, Excel, and JSON format support
- **Price Alerts**: Set notifications for price movements
- **Auto-refresh**: Automatic data updates at configurable intervals
- **Memory Optimization**: Intelligent cache management and cleanup
- **Performance Monitoring**: Real-time statistics and usage tracking

### Enterprise Features
- **Command-line Interface**: Professional CLI with comprehensive options
- **Backup System**: Automated backups with metadata and rotation
- **Settings Management**: User-friendly configuration interface
- **Theme Support**: Light and dark theme options
- **Comprehensive Testing**: Full test coverage with automated test suite
- **Memory Management**: Advanced optimization and monitoring

### API Architecture
- **Node.js Backend**: Express API with Prisma ORM for data persistence
- **Dart Client**: Type-safe API client for Flutter integration
- **Dashboard API**: Single aggregated payload endpoint for UI consumption
- **Database System**: Microsoft SQL Server with Prisma ORM for enterprise-grade data management

## 🔧 CI/CD Pipeline

This project uses GitHub Actions for automated testing and deployment.

### Workflows

| Workflow | Purpose | Trigger |
|----------|---------|---------|
| **Main CI/CD** | Build, test, deploy | Push to `master` |
| **Deploy Pages** | Web dashboard deployment | Push to `master` |
| **PR Checks** | Validate PRs | Pull requests |

### Automated Testing

- **Dart Tests**: Formatting, analysis, unit tests with coverage
- **Node.js Tests**: Linting, unit tests, integration tests with SQL Server
- **Security Scan**: Vulnerability scanning with Trivy

### Deployment

- **GitHub Pages**: Automated web dashboard deployment on every push to `master`
- **Live URL**: [https://getuser-shivam.github.io/NEPSE-Analysis/](https://getuser-shivam.github.io/NEPSE-Analysis/)

### Setup Required Secrets

Configure these in GitHub repository settings:

- `GROQ_API_KEY` - Groq AI API key
- `POLLENS_API_KEY` - Pollens AI API key

## 📱 Mobile Development

### Android Debugging

The NEPSE Analysis Flutter app supports USB and wireless debugging.

**Quick Setup:**

```bash
# Navigate to Flutter app
cd apps/mobile

# Install dependencies
flutter pub get

# Run on connected device
flutter run
```

**VS Code Launch Configurations:**

1. Open **Run and Debug** panel (Ctrl+Shift+D)
2. Select configuration:
   - `NEPSE Analysis (USB Device)` - For USB debugging
   - `NEPSE Analysis (Wireless)` - For wireless debugging
   - `NEPSE Analysis (Profile Mode)` - For performance testing

## 📋 Installation

### Prerequisites
- **Dart SDK** >= 3.11.0
- **Flutter** (stable channel)
- **Node.js** >= 20.x
- **MySQL** (for backend database)
- **npm** package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/getuser-shivam/NEPSE-Analysis.git
cd NEPSE-Analysis

# Backend setup
cd apps/api
npm install
npx prisma generate
npm run dev

# Mobile app setup (in a new terminal)
cd apps/mobile
flutter pub get
flutter run

# Web client setup (in a new terminal)
cd packages/client_sdk
dart pub get
```

### Dependencies

**Backend (Node.js):**
- Express >= 4.x
- Prisma ORM >= 6.x
- Zod (validation)
- Groq SDK (AI integration)

**Mobile (Flutter/Dart):**
- flutter_riverpod (state management)
- fl_chart (charting)
- google_fonts (typography)
- http (networking)

**Web Client (Dart):**
- http (API communication)

## 🧪 Testing

### Run Dart Tests
```bash
cd apps/mobile
flutter test
```

### Run Node.js Tests
```bash
cd apps/api
npm test
```

### Test Categories
- **Unit Tests**: Individual function and method testing (`test/unit/`)
- **Widget Tests**: UI component testing (`test/widget/`)
- **Integration Tests**: Component interaction testing (`test/integration/`)
- **E2E Tests**: End-to-end application testing (`test/e2e/`)

## 📈 Technical Indicators

### RSI (Relative Strength Index)
- **Purpose**: Momentum oscillator measuring overbought/oversold conditions
- **Range**: 0-100, with 70+ indicating overbought, 30- indicating oversold
- **Period**: Default 14 days

### MACD (Moving Average Convergence Divergence)
- **Purpose**: Trend-following momentum indicator
- **Components**: MACD line, signal line, histogram
- **Periods**: Fast (12), Slow (26), Signal (9)

### Bollinger Bands
- **Purpose**: Volatility measurement and trend identification
- **Components**: Upper band, middle band (SMA), lower band
- **Period**: Default 20 days, 2 standard deviations

### Stochastic Oscillator
- **Purpose**: Momentum indicator comparing closing price to price range
- **Range**: 0-100, with 80+ overbought, 20- oversold
- **Periods**: %K (14), %D (3)

### Williams %R
- **Purpose**: Momentum indicator similar to Stochastic
- **Range**: -100 to 0, with -20 overbought, -80 oversold
- **Period**: Default 14 days

## 📁 Project Structure

```
NEPSE-Analysis/
├── apps/
│   ├── api/                    # Node.js + Prisma + Express Backend
│   │   ├── prisma/             # Database schema & migrations (MySQL)
│   │   ├── src/                # API routes, controllers, services
│   │   └── package.json
│   └── mobile/                 # Flutter Mobile Application
│       ├── lib/                # Dart source code
│       │   ├── main.dart       # Application entry point
│       │   ├── models/         # Data models
│       │   ├── providers/      # Riverpod state management
│       │   ├── screens/        # UI screens
│       │   ├── services/       # Business logic
│       │   ├── theme/          # App theming
│       │   ├── utils.dart      # Technical indicator calculations
│       │   └── widgets/        # Reusable UI components
│       ├── test/               # Test suite
│       │   ├── unit/           # Unit tests
│       │   ├── widget/         # Widget tests
│       │   ├── integration/    # Integration tests
│       │   └── e2e/            # End-to-end tests
│       └── web/                # Flutter web build
├── packages/
│   └── client_sdk/             # Dart Web Client & Dashboard
│       ├── lib/                # SDK source code
│       ├── web/                # Web dashboard (deployed to GitHub Pages)
│       │   ├── index.html      # Dashboard entry point
│       │   ├── style.css       # Glassmorphic styles
│       │   └── main.dart.js    # Compiled Dart application
│       └── test/               # SDK tests
├── .github/
│   ├── workflows/              # CI/CD pipeline definitions
│   └── dependabot.yml          # Automated dependency updates
└── tools/                      # Utility scripts
```

### Stack Components

- **Backend (Node.js)**: Enterprise-ready API with Prisma ORM and SQL Server.
- **Mobile (Flutter/Dart)**: Cross-platform mobile app with Riverpod state management.
- **Web Client (Dart)**: Lightweight web dashboard and SDK for NEPSE data.
- **Database (SQL Server)**: Microsoft SQL Server for production-grade relational data persistence.

## 🔧 Configuration

### Environment Variables

Create a `.env` file in `apps/api/`:
```env
DATABASE_URL="sqlserver://localhost:1433;database=nepse_analysis;user=sa;password=YourStrong@Passw0rd;trustServerCertificate=true"
GROQ_API_KEY="your-groq-api-key"
POLLENS_API_KEY="your-pollens-api-key"
PORT=3000
```

### Database Setup

```bash
cd apps/api

# Generate Prisma client
npx prisma generate

# Push schema to SQL Server
npx prisma db push

# Seed initial data
npm run seed

# Open database GUI
npx prisma studio
```

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/getuser-shivam/NEPSE-Analysis.git
cd NEPSE-Analysis

# Install all dependencies
cd apps/api && npm install && cd ../..
cd apps/mobile && flutter pub get && cd ../..
cd packages/client_sdk && dart pub get && cd ../..

# Run tests
cd apps/mobile && flutter test
cd apps/api && npm test
```

### Code Style
- Follow Dart [Effective Dart](https://dart.dev/effective-dart) guidelines
- Use `dart format` for consistent formatting
- Add comprehensive documentation comments
- Write tests for new features
- Maintain test coverage

### Submitting Changes
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request to `master`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NEPSE**: Nepal Stock Exchange for market data
- **Yahoo Finance**: Additional data source and API
- **Flutter**: Cross-platform UI framework
- **Prisma**: Next-generation ORM for Node.js
- **ApexCharts**: Interactive chart visualization
- **Riverpod**: Reactive state management for Flutter

## 📞 Support

For support, questions, or contributions:
- **Issues**: [GitHub Issues](https://github.com/getuser-shivam/NEPSE-Analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/getuser-shivam/NEPSE-Analysis/discussions)

---

**NEPSE Analysis Tool** - Professional stock analysis for the Nepal Stock Exchange

*Built with ❤️ for Nepali investors*
