#!/usr/bin/env node

/**
 * Build script for NEPSE Analysis Dashboard
 * Generates static web assets for GitHub Pages deployment
 */

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ROOT_DIR = path.join(__dirname, '..', '..');
const SITE_DIR = path.join(ROOT_DIR, '_site');

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  blue: '\x1b[34m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

async function ensureDir(dir) {
  try {
    await fs.mkdir(dir, { recursive: true });
  } catch (error) {
    log(`Error creating directory ${dir}: ${error.message}`, 'red');
    throw error;
  }
}

async function copyFile(src, dest) {
  try {
    await fs.copyFile(src, dest);
    log(`✓ Copied: ${path.basename(src)}`, 'green');
  } catch (error) {
    log(`✗ Failed to copy ${src}: ${error.message}`, 'red');
  }
}

async function copyDirectory(src, dest) {
  try {
    await ensureDir(dest);
    const entries = await fs.readdir(src, { withFileTypes: true });
    
    for (const entry of entries) {
      const srcPath = path.join(src, entry.name);
      const destPath = path.join(dest, entry.name);
      
      if (entry.isDirectory()) {
        await copyDirectory(srcPath, destPath);
      } else {
        await copyFile(srcPath, destPath);
      }
    }
    log(`✓ Copied directory: ${path.basename(src)}`, 'green');
  } catch (error) {
    log(`✗ Failed to copy directory ${src}: ${error.message}`, 'red');
  }
}

async function generateIndexHtml() {
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="NEPSE Analysis - Stock market analysis tool for Nepal Stock Exchange">
    <meta name="keywords" content="NEPSE, Nepal Stock Exchange, stock analysis, technical indicators, portfolio management">
    <meta name="author" content="Shivam">
    <meta name="theme-color" content="#1a237e">
    
    <!-- Open Graph / Social Media -->
    <meta property="og:title" content="NEPSE Analysis Tool">
    <meta property="og:description" content="Advanced stock analysis for Nepal Stock Exchange with AI-powered insights">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://getuser-shivam.github.io/NEPSE-Analysis/">
    
    <!-- Security Headers (meta equivalent) -->
    <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; connect-src 'self' https://api.github.com;">
    <meta http-equiv="X-Content-Type-Options" content="nosniff">
    <meta http-equiv="X-Frame-Options" content="DENY">
    <meta http-equiv="X-XSS-Protection" content="1; mode=block">
    <meta http-equiv="Referrer-Policy" content="strict-origin-when-cross-origin">
    
    <title>NEPSE Analysis Tool</title>
    
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    
    <style>
        body { font-family: 'Inter', sans-serif; }
        .gradient-bg { background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%); }
        .card-hover { transition: all 0.3s ease; }
        .card-hover:hover { transform: translateY(-4px); box-shadow: 0 10px 40px rgba(0,0,0,0.1); }
        .fade-in { animation: fadeIn 0.6s ease-in; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body class="bg-gray-50 text-gray-800">
    <!-- Header -->
    <header class="gradient-bg text-white py-6 shadow-lg">
        <div class="container mx-auto px-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center space-x-3">
                    <i data-lucide="trending-up" class="w-8 h-8"></i>
                    <div>
                        <h1 class="text-2xl font-bold">NEPSE Analysis</h1>
                        <p class="text-blue-200 text-sm">Advanced Stock Market Analytics</p>
                    </div>
                </div>
                <nav class="hidden md:flex space-x-6">
                    <a href="#features" class="hover:text-blue-200 transition">Features</a>
                    <a href="#docs" class="hover:text-blue-200 transition">Documentation</a>
                    <a href="https://github.com/getuser-shivam/NEPSE-Analysis" target="_blank" class="hover:text-blue-200 transition flex items-center gap-1">
                        <i data-lucide="github" class="w-4 h-4"></i> GitHub
                    </a>
                </nav>
            </div>
        </div>
    </header>

    <!-- Hero Section -->
    <section class="py-20 gradient-bg text-white">
        <div class="container mx-auto px-4 text-center">
            <h2 class="text-4xl md:text-5xl font-bold mb-6 fade-in">
                Analyze NEPSE Stocks with AI-Powered Insights
            </h2>
            <p class="text-xl text-blue-100 mb-8 max-w-2xl mx-auto fade-in" style="animation-delay: 0.1s;">
                Comprehensive technical analysis, portfolio management, and machine learning predictions for Nepal Stock Exchange
            </p>
            <div class="flex flex-wrap justify-center gap-4 fade-in" style="animation-delay: 0.2s;">
                <a href="https://github.com/getuser-shivam/NEPSE-Analysis" class="bg-white text-blue-900 px-8 py-3 rounded-lg font-semibold hover:bg-blue-50 transition flex items-center gap-2">
                    <i data-lucide="download" class="w-5 h-5"></i>
                    Get Started
                </a>
                <a href="#docs" class="border-2 border-white text-white px-8 py-3 rounded-lg font-semibold hover:bg-white hover:text-blue-900 transition flex items-center gap-2">
                    <i data-lucide="book-open" class="w-5 h-5"></i>
                    Documentation
                </a>
            </div>
        </div>
    </section>

    <!-- Features Grid -->
    <section id="features" class="py-16">
        <div class="container mx-auto px-4">
            <h3 class="text-3xl font-bold text-center mb-12">Key Features</h3>
            <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                <!-- Feature 1 -->
                <div class="bg-white rounded-xl p-6 shadow-md card-hover">
                    <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                        <i data-lucide="activity" class="w-6 h-6 text-blue-600"></i>
                    </div>
                    <h4 class="text-xl font-semibold mb-2">Technical Indicators</h4>
                    <p class="text-gray-600">RSI, MACD, Bollinger Bands, Stochastic, and Williams %R with real-time calculations</p>
                </div>

                <!-- Feature 2 -->
                <div class="bg-white rounded-xl p-6 shadow-md card-hover">
                    <div class="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
                        <i data-lucide="pie-chart" class="w-6 h-6 text-green-600"></i>
                    </div>
                    <h4 class="text-xl font-semibold mb-2">Portfolio Management</h4>
                    <p class="text-gray-600">Track investments, calculate gains/losses, and analyze portfolio performance metrics</p>
                </div>

                <!-- Feature 3 -->
                <div class="bg-white rounded-xl p-6 shadow-md card-hover">
                    <div class="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
                        <i data-lucide="brain" class="w-6 h-6 text-purple-600"></i>
                    </div>
                    <h4 class="text-xl font-semibold mb-2">AI-Powered Analysis</h4>
                    <p class="text-gray-600">Groq and Pollens AI integration for predictive modeling and trend analysis</p>
                </div>

                <!-- Feature 4 -->
                <div class="bg-white rounded-xl p-6 shadow-md card-hover">
                    <div class="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center mb-4">
                        <i data-lucide="bell" class="w-6 h-6 text-orange-600"></i>
                    </div>
                    <h4 class="text-xl font-semibold mb-2">Price Alerts</h4>
                    <p class="text-gray-600">Set notifications for price movements, percentage changes, and indicator signals</p>
                </div>

                <!-- Feature 5 -->
                <div class="bg-white rounded-xl p-6 shadow-md card-hover">
                    <div class="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center mb-4">
                        <i data-lucide="eye" class="w-6 h-6 text-red-600"></i>
                    </div>
                    <h4 class="text-xl font-semibold mb-2">Watchlist</h4>
                    <p class="text-gray-600">Monitor multiple stocks simultaneously with customizable watchlists</p>
                </div>

                <!-- Feature 6 -->
                <div class="bg-white rounded-xl p-6 shadow-md card-hover">
                    <div class="w-12 h-12 bg-teal-100 rounded-lg flex items-center justify-center mb-4">
                        <i data-lucide="database" class="w-6 h-6 text-teal-600"></i>
                    </div>
                    <h4 class="text-xl font-semibold mb-2">Data Export</h4>
                    <p class="text-gray-600">Export data in CSV, Excel, and JSON formats for further analysis</p>
                </div>
            </div>
        </div>
    </section>

    <!-- Stats Section -->
    <section class="py-16 bg-gray-100">
        <div class="container mx-auto px-4">
            <div class="grid md:grid-cols-4 gap-8 text-center">
                <div>
                    <div class="text-4xl font-bold text-blue-600 mb-2">23+</div>
                    <div class="text-gray-600">Unit Tests</div>
                </div>
                <div>
                    <div class="text-4xl font-bold text-green-600 mb-2">10+</div>
                    <div class="text-gray-600">Technical Indicators</div>
                </div>
                <div>
                    <div class="text-4xl font-bold text-purple-600 mb-2">3</div>
                    <div class="text-gray-600">Platform Support</div>
                </div>
                <div>
                    <div class="text-4xl font-bold text-orange-600 mb-2">2</div>
                    <div class="text-gray-600">AI Services</div>
                </div>
            </div>
        </div>
    </section>

    <!-- Documentation Section -->
    <section id="docs" class="py-16">
        <div class="container mx-auto px-4">
            <h3 class="text-3xl font-bold text-center mb-12">Documentation</h3>
            <div class="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
                <a href="https://github.com/getuser-shivam/NEPSE-Analysis#readme" class="bg-white rounded-xl p-6 shadow-md card-hover block">
                    <div class="flex items-center gap-3 mb-3">
                        <i data-lucide="file-text" class="w-6 h-6 text-blue-600"></i>
                        <h4 class="text-xl font-semibold">README</h4>
                    </div>
                    <p class="text-gray-600">Installation, configuration, and usage instructions</p>
                </a>

                <a href="./dart-docs/" class="bg-white rounded-xl p-6 shadow-md card-hover block">
                    <div class="flex items-center gap-3 mb-3">
                        <i data-lucide="code" class="w-6 h-6 text-green-600"></i>
                        <h4 class="text-xl font-semibold">API Reference</h4>
                    </div>
                    <p class="text-gray-600">Dart API documentation for developers</p>
                </a>

                <a href="https://github.com/getuser-shivam/NEPSE-Analysis/blob/main/CI_CD_GUIDE.md" class="bg-white rounded-xl p-6 shadow-md card-hover block">
                    <div class="flex items-center gap-3 mb-3">
                        <i data-lucide="git-branch" class="w-6 h-6 text-purple-600"></i>
                        <h4 class="text-xl font-semibold">CI/CD Guide</h4>
                    </div>
                    <p class="text-gray-600">GitHub Actions workflows and deployment guide</p>
                </a>

                <a href="https://github.com/getuser-shivam/NEPSE-Analysis/blob/main/CHANGELOG.md" class="bg-white rounded-xl p-6 shadow-md card-hover block">
                    <div class="flex items-center gap-3 mb-3">
                        <i data-lucide="history" class="w-6 h-6 text-orange-600"></i>
                        <h4 class="text-xl font-semibold">Changelog</h4>
                    </div>
                    <p class="text-gray-600">Version history and release notes</p>
                </a>
            </div>
        </div>
    </section>

    <!-- Tech Stack -->
    <section class="py-16 bg-gray-100">
        <div class="container mx-auto px-4">
            <h3 class="text-3xl font-bold text-center mb-12">Technology Stack</h3>
            <div class="flex flex-wrap justify-center gap-6">
                <span class="px-4 py-2 bg-white rounded-full shadow text-sm font-medium">Python</span>
                <span class="px-4 py-2 bg-white rounded-full shadow text-sm font-medium">Dart</span>
                <span class="px-4 py-2 bg-white rounded-full shadow text-sm font-medium">Node.js</span>
                <span class="px-4 py-2 bg-white rounded-full shadow text-sm font-medium">Prisma</span>
                <span class="px-4 py-2 bg-white rounded-full shadow text-sm font-medium">SQL Server</span>
                <span class="px-4 py-2 bg-white rounded-full shadow text-sm font-medium">Flutter</span>
                <span class="px-4 py-2 bg-white rounded-full shadow text-sm font-medium">Tailwind CSS</span>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="bg-gray-900 text-white py-12">
        <div class="container mx-auto px-4">
            <div class="flex flex-col md:flex-row justify-between items-center">
                <div class="mb-4 md:mb-0">
                    <h5 class="text-xl font-bold mb-2">NEPSE Analysis</h5>
                    <p class="text-gray-400">Professional stock analysis for Nepal Stock Exchange</p>
                </div>
                <div class="flex space-x-6">
                    <a href="https://github.com/getuser-shivam/NEPSE-Analysis" target="_blank" class="text-gray-400 hover:text-white transition">
                        <i data-lucide="github" class="w-6 h-6"></i>
                    </a>
                    <a href="mailto:shivam@example.com" class="text-gray-400 hover:text-white transition">
                        <i data-lucide="mail" class="w-6 h-6"></i>
                    </a>
                </div>
            </div>
            <div class="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400 text-sm">
                <p>&copy; 2024 NEPSE Analysis. Built with ❤️ for Nepali investors.</p>
                <p class="mt-2">Deployed on GitHub Pages with SSL/HTTPS</p>
            </div>
        </div>
    </footer>

    <!-- Initialize Lucide Icons -->
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            lucide.createIcons();
        });
    </script>
</body>
</html>`;

  try {
    await fs.writeFile(path.join(SITE_DIR, 'index.html'), html);
    log('✓ Generated index.html', 'green');
  } catch (error) {
    log(`✗ Failed to generate index.html: ${error.message}`, 'red');
    throw error;
  }
}

async function generate404Html() {
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>404 - Page Not Found | NEPSE Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        .gradient-bg { background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%); }
    </style>
</head>
<body class="bg-gray-50">
    <div class="min-h-screen flex items-center justify-center">
        <div class="text-center">
            <h1 class="text-6xl font-bold text-blue-900 mb-4">404</h1>
            <h2 class="text-2xl font-semibold text-gray-700 mb-4">Page Not Found</h2>
            <p class="text-gray-600 mb-8">The page you're looking for doesn't exist.</p>
            <a href="/NEPSE-Analysis/" class="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition">
                Go Home
            </a>
        </div>
    </div>
</body>
</html>`;

  try {
    await fs.writeFile(path.join(SITE_DIR, '404.html'), html);
    log('✓ Generated 404.html', 'green');
  } catch (error) {
    log(`✗ Failed to generate 404.html: ${error.message}`, 'red');
  }
}

async function copyAssets() {
  const sourceDirs = [
    { src: path.join(ROOT_DIR, 'nepse_analysis', 'web'), dest: path.join(SITE_DIR, 'dart-client') },
    { src: path.join(ROOT_DIR, 'nepse_app', 'build', 'web'), dest: path.join(SITE_DIR, 'flutter-app') },
  ];

  for (const { src, dest } of sourceDirs) {
    try {
      await fs.access(src);
      await copyDirectory(src, dest);
    } catch {
      log(`⚠ Skipped: ${path.basename(src)} (not found)`, 'yellow');
    }
  }
}

async function generateCNAME() {
  const cname = 'getuser-shivam.github.io';
  try {
    await fs.writeFile(path.join(SITE_DIR, 'CNAME'), cname);
    log('✓ Generated CNAME file', 'green');
  } catch (error) {
    log(`✗ Failed to generate CNAME: ${error.message}`, 'red');
  }
}

async function generateRobotsTxt() {
  const robots = `User-agent: *
Allow: /

Sitemap: https://getuser-shivam.github.io/NEPSE-Analysis/sitemap.xml
`;
  try {
    await fs.writeFile(path.join(SITE_DIR, 'robots.txt'), robots);
    log('✓ Generated robots.txt', 'green');
  } catch (error) {
    log(`✗ Failed to generate robots.txt: ${error.message}`, 'red');
  }
}

async function main() {
  log('\n🏗️  Building NEPSE Analysis Dashboard...\n', 'blue');

  try {
    // Ensure site directory exists
    await ensureDir(SITE_DIR);

    // Generate HTML files
    await generateIndexHtml();
    await generate404Html();

    // Copy assets
    await copyAssets();

    // Generate config files
    await generateCNAME();
    await generateRobotsTxt();

    // Copy documentation
    const docs = ['README.md', 'CHANGELOG.md', 'CI_CD_GUIDE.md'];
    for (const doc of docs) {
      const src = path.join(ROOT_DIR, doc);
      const dest = path.join(SITE_DIR, doc);
      try {
        await fs.access(src);
        await copyFile(src, dest);
      } catch {
        log(`⚠ Skipped: ${doc} (not found)`, 'yellow');
      }
    }

    log('\n✅ Build completed successfully!', 'green');
    log(`📁 Output directory: ${SITE_DIR}\n`, 'blue');

    // Print file list
    const files = await fs.readdir(SITE_DIR, { recursive: true });
    log('Generated files:', 'blue');
    for (const file of files) {
      log(`  - ${file}`, 'reset');
    }

  } catch (error) {
    log(`\n❌ Build failed: ${error.message}`, 'red');
    process.exit(1);
  }
}

main();
