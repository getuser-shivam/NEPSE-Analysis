import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../theme/app_theme.dart';
import '../widgets/common/glass_container.dart';
import '../widgets/common/status_pill.dart';
import '../widgets/dashboard/nepse_market_chart.dart';
import '../providers/dashboard_provider.dart';
import '../providers/api_client_provider.dart';
import 'package:nepse_client/nepse_client.dart';
import 'dart:ui';
import 'dart:math' as math;

class DashboardScreen extends ConsumerWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final dashboardAsync = ref.watch(dashboardProvider);

    return Scaffold(
      body: DecoratedBox(
        decoration: const BoxDecoration(gradient: AppTheme.backgroundGradient),
        child: CustomScrollView(
          slivers: [
            _buildSliverAppBar(context, ref),
            SliverPadding(
              padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 16.0),
              sliver: dashboardAsync.when(
                loading: () => const SliverFillRemaining(
                  child: Center(
                    child: CircularProgressIndicator(color: AppColors.accent),
                  ),
                ),
                error: (error, stack) => SliverFillRemaining(
                  child: Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(Icons.error_outline, color: AppColors.negative, size: 64),
                        const SizedBox(height: 16),
                        Text('Backend Offline: $error', style: const TextStyle(color: AppColors.negative)),
                        const SizedBox(height: 24),
                        ElevatedButton(
                          onPressed: () => ref.refresh(dashboardProvider),
                          child: const Text('Retry Connection'),
                        )
                      ],
                    ),
                  ),
                ),
                data: (snapshot) {
                  // Fallback for empty datastores
                  if (snapshot.symbols.isEmpty) {
                    return const SliverFillRemaining(child: Center(child: Text("Waiting for market data...")));
                  }
                  
                  // Sort symbols to find top gainers
                  final sortedSymbols = List<DashboardSymbolSnapshot>.from(snapshot.symbols);
                  sortedSymbols.sort((a, b) {
                    if (a.recentPrices.isEmpty || b.recentPrices.isEmpty) return 0;
                    double aPct = (a.recentPrices.first.close - a.recentPrices.first.open) / a.recentPrices.first.open;
                    double bPct = (b.recentPrices.first.close - b.recentPrices.first.open) / b.recentPrices.first.open;
                    return bPct.compareTo(aPct); // descending
                  });

                  final mainSymbol = snapshot.symbols.first;
                  final rawPrices = mainSymbol.recentPrices.toList().reversed.toList();
                  final prices = rawPrices.map((e) => e.close).toList();
                  final dates = rawPrices.map((e) => e.tradeDate.toString().split(' ')[0]).toList();

                  return SliverList(
                    delegate: SliverChildListDelegate([
                      // Main Pulse Row
                      LayoutBuilder(
                        builder: (context, constraints) {
                          if (constraints.maxWidth > 800) {
                            return Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Expanded(flex: 3, child: _buildMarketPulse(mainSymbol.symbol, prices, dates)),
                                const SizedBox(width: 24),
                                Expanded(flex: 2, child: _buildTopGainers(sortedSymbols, constraints)),
                              ],
                            );
                          }
                          return Column(
                            children: [
                               _buildMarketPulse(mainSymbol.symbol, prices, dates),
                               const SizedBox(height: 24),
                               _buildTopGainers(sortedSymbols, constraints),
                            ],
                          );
                        }
                      ),
                      const SizedBox(height: 24),
                      _buildQuickStats(),
                    ]),
                  );
                },
              )
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSliverAppBar(BuildContext context, WidgetRef ref) {
    return SliverAppBar(
      expandedHeight: 80, // Slimmer for web
      floating: true,
      pinned: true,
      flexibleSpace: FlexibleSpaceBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.auto_graph_rounded, color: AppColors.accent, size: 24),
            const SizedBox(width: 8),
            const Text('NEPSE Hub'),
          ],
        ),
        titlePadding: const EdgeInsets.only(left: 24, bottom: 16),
      ),
      actions: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 8.0),
          child: ElevatedButton.icon(
            onPressed: () => _showAuthModal(context, ref),
            icon: const Icon(Icons.security, size: 18),
            label: const Text('Secure Login'),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.accent.withValues(alpha: 0.15),
              foregroundColor: AppColors.accent,
              shadowColor: AppColors.accentGlow,
              elevation: 8,
              side: BorderSide(color: AppColors.accent.withValues(alpha: 0.5)),
            ),
          ),
        ),
      ],
    );
  }

  // --- NATIVE FLUTTER GLASSMORPHIC AUTH DIALOG ---
  void _showAuthModal(BuildContext context, WidgetRef ref) {
    final emailController = TextEditingController();
    final passwordController = TextEditingController();
    bool isAuthenticating = false;

    showGeneralDialog(
      context: context,
      barrierDismissible: true,
      barrierLabel: 'Dismiss',
      transitionDuration: const Duration(milliseconds: 300),
      pageBuilder: (context, anim1, anim2) {
        return StatefulBuilder(builder: (context, setState) {
          return BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 30, sigmaY: 30),
            child: Center(
              child: Material(
                color: Colors.transparent,
                child: GlassContainer(
                  padding: const EdgeInsets.all(40),
                  width: 420,
                  gradientColors: [
                    const Color(0xFF0F1721).withValues(alpha: 0.65),
                    const Color(0xFF0F1721).withValues(alpha: 0.45)
                  ],
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            'Secure Access',
                            style: TextStyle(
                              fontSize: 28,
                              fontWeight: FontWeight.w800,
                              letterSpacing: -0.5,
                              color: Colors.white,
                            ),
                          ),
                          IconButton(
                            icon: const Icon(Icons.close, color: AppColors.textSecondary),
                            onPressed: () => Navigator.pop(context),
                          )
                        ],
                      ),
                      const SizedBox(height: 16),
                      const Text(
                        'Sign in natively via Flutter SDK to securely sync live metrics from our Prisma backend.',
                        style: TextStyle(color: AppColors.textSecondary, height: 1.5),
                      ),
                      const SizedBox(height: 32),
                      TextField(
                        controller: emailController,
                        decoration: InputDecoration(
                          labelText: 'Phone / Email',
                          filled: true,
                          fillColor: Colors.black.withValues(alpha: 0.3),
                        ),
                      ),
                      const SizedBox(height: 20),
                      TextField(
                        controller: passwordController,
                        obscureText: true,
                        decoration: InputDecoration(
                          labelText: 'Password',
                          filled: true,
                          fillColor: Colors.black.withValues(alpha: 0.3),
                        ),
                      ),
                      const SizedBox(height: 32),
                      SizedBox(
                        width: double.infinity,
                        height: 56,
                        child: ElevatedButton(
                          onPressed: isAuthenticating ? null : () async {
                            setState(() => isAuthenticating = true);
                            try {
                              final secureStorage = ref.read(secureStorageProvider);
                              final payload = await ref.read(apiClientProvider).login(
                                emailController.text, 
                                passwordController.text
                              );
                              
                              // Preserve JWT seamlessly into mobile KeyChain/Keystore natively
                              if (payload.containsKey('accessToken')) {
                                await secureStorage.write(key: 'jwt_token', value: payload['accessToken']);
                              }
                              
                              if (context.mounted) {
                                Navigator.pop(context);
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(backgroundColor: AppColors.positive, content: Text('Authentication Successful!')),
                                );
                                ref.refresh(dashboardProvider); // Reload Data With Auth Headers
                              }
                            } catch (e) {
                              setState(() => isAuthenticating = false);
                              if (context.mounted) {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  SnackBar(backgroundColor: AppColors.negative, content: Text('Error: $e')),
                                );
                              }
                            }
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppColors.accent,
                            foregroundColor: Colors.black,
                          ),
                          child: isAuthenticating 
                              ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(color: Colors.black, strokeWidth: 2))
                              : const Text('Authenticate', style: TextStyle(fontSize: 16)),
                        ),
                      )
                    ],
                  ),
                ),
              ),
            ),
          );
        });
      },
      transitionBuilder: (context, anim1, anim2, child) {
        return Transform.scale(
          scale: 0.95 + (0.05 * anim1.value),
          child: FadeTransition(
            opacity: anim1,
            child: child,
          ),
        );
      },
    );
  }

  Widget _buildMarketPulse(String marketName, List<double> mockPrices, List<String> mockDates) {
    // Stateful Builder handles the local toggle state for the SMA chart overlay
    return StatefulBuilder(
      builder: (context, setState) {
        bool showSMA = false; 
        bool showMACD = false;
        
        // Calculate recent change using latest vs previous real data point
        double pctChange = 0;
        double amtChange = 0;
        if (mockPrices.length > 2) {
            amtChange = mockPrices.last - mockPrices[mockPrices.length - 2];
            pctChange = (amtChange / mockPrices[mockPrices.length - 2]) * 100;
        }

        return GlassContainer(
          padding: const EdgeInsets.all(30),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        marketName,
                        style: const TextStyle(color: AppColors.textSecondary, fontSize: 16),
                      ),
                      const SizedBox(height: 4),
                      Row(
                        children: [
                          Text(
                            mockPrices.last.toStringAsFixed(2),
                            style: const TextStyle(fontSize: 42, fontWeight: FontWeight.w900, letterSpacing: -1),
                          ),
                          const SizedBox(width: 16),
                          StatusPill(
                            label: '${amtChange >= 0 ? "+" : ""}${amtChange.toStringAsFixed(2)} (${pctChange.toStringAsFixed(2)}%)', 
                            color: amtChange >= 0 ? AppColors.positive : AppColors.negative
                          ),
                        ],
                      ),
                    ],
                  ),
                  // Chart Control Toggles
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(4),
                        decoration: BoxDecoration(
                          color: Colors.black.withValues(alpha: 0.3),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
                        ),
                        child: FilterChip(
                          label: const Text('SMA 5', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 12)),
                          selected: showSMA,
                          onSelected: (val) => setState(() => showSMA = val),
                          selectedColor: AppColors.accent.withValues(alpha: 0.2),
                          checkmarkColor: AppColors.accent,
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.all(4),
                        decoration: BoxDecoration(
                          color: Colors.black.withValues(alpha: 0.3),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
                        ),
                        child: FilterChip(
                          label: const Text('MACD', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 12)),
                          selected: showMACD,
                          onSelected: (val) => setState(() => showMACD = val),
                          selectedColor: Colors.blueAccent.withValues(alpha: 0.2),
                          checkmarkColor: Colors.blueAccent,
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                        ),
                      ),
                    ],
                  )
                ],
              ),
              const SizedBox(height: 32),
              // Chart Render Area
              SizedBox(
                height: showMACD ? 450 : 350,
                child: NepseMarketChart(
                  prices: mockPrices,
                  dates: mockDates,
                  showSMA: showSMA,
                  showMACD: showMACD,
                ),
              ),
            ],
          ),
        );
      }
    );
  }

  Widget _buildTopGainers(List<DashboardSymbolSnapshot> gainers, BoxConstraints constraints) {
    return GlassContainer(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Top Movers', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: DataTable(
              headingTextStyle: const TextStyle(color: AppColors.textSecondary, fontWeight: FontWeight.w600),
              dataTextStyle: const TextStyle(color: Colors.white, fontSize: 16),
              columnSpacing: constraints.maxWidth > 800 ? 56 : 24,
              columns: const [
                DataColumn(label: Text('Symbol')),
                DataColumn(label: Text('LTP')),
                DataColumn(label: Text('Change')),
              ],
              rows: gainers.take(10).map((gainer) {
                  final prices = gainer.recentPrices;
                  if (prices.isEmpty) return const DataRow(cells: [DataCell(Text('')), DataCell(Text('')), DataCell(Text(''))]);
                  
                  final ltp = prices.first.close;
                  final amtChange = prices.first.close - prices.first.open;
                  final pctChange = (amtChange / prices.first.open) * 100;
                  final isPositive = amtChange >= 0;

                  return DataRow(cells: [
                     DataCell(Text(gainer.symbol, style: const TextStyle(fontWeight: FontWeight.bold))),
                     DataCell(Text(ltp.toStringAsFixed(2))),
                     DataCell(Text('${isPositive ? '+' : ''}${amtChange.toStringAsFixed(2)} (${pctChange.toStringAsFixed(2)}%)', style: TextStyle(color: isPositive ? AppColors.positive : AppColors.negative))),
                  ]);
              }).toList()
            ),
          )
        ],
      )
    );
  }

  Widget _buildQuickStats() {
    return Row(
      children: [
        Expanded(
          child: GlassContainer(
            padding: const EdgeInsets.all(16),
            child: const Column(
              children: [
                Text(
                  'Advancing',
                  style: TextStyle(
                    fontSize: 12,
                    color: AppColors.textSecondary,
                  ),
                ),
                Text(
                  '142',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: AppColors.positive,
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: GlassContainer(
            padding: const EdgeInsets.all(16),
            child: const Column(
              children: [
                Text(
                  'Declining',
                  style: TextStyle(
                    fontSize: 12,
                    color: AppColors.textSecondary,
                  ),
                ),
                Text(
                  '85',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: AppColors.negative,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  // Quick stats is kept but re-styled identically
}
