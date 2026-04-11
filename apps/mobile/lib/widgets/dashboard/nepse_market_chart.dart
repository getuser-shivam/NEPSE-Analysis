import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import '../../theme/app_theme.dart';

class NepseMarketChart extends StatefulWidget {
  final List<double> prices;
  final List<String> dates;
  final bool showSMA;
  final bool showMACD;

  const NepseMarketChart({
    super.key,
    required this.prices,
    required this.dates,
    this.showSMA = false,
    this.showMACD = false,
  });

  @override
  State<NepseMarketChart> createState() => _NepseMarketChartState();
}

class _NepseMarketChartState extends State<NepseMarketChart> {
  List<double> _smaData = [];
  List<double> _macdLine = [];
  List<double> _macdSignal = [];
  List<double> _macdHistogram = [];

  @override
  void initState() {
    super.initState();
    _calculateIndicators();
  }

  @override
  void didUpdateWidget(NepseMarketChart oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.prices != widget.prices || oldWidget.showSMA != widget.showSMA || oldWidget.showMACD != widget.showMACD) {
      _calculateIndicators();
    }
  }

  void _calculateIndicators() {
    // 1. SMA Engine
    if (widget.showSMA && widget.prices.isNotEmpty) {
      const period = 5;
      _smaData = List<double>.filled(widget.prices.length, 0);
      
      for (int i = 0; i < widget.prices.length; i++) {
        if (i < period - 1) {
          _smaData[i] = widget.prices[i];
        } else {
          double sum = 0;
          for (int j = 0; j < period; j++) {
            sum += widget.prices[i - j];
          }
          _smaData[i] = sum / period;
        }
      }
    } else {
      _smaData = [];
    }

    // 2. MACD Deep Algorithm Engine (EMA12, EMA26, Signal9)
    if (widget.showMACD && widget.prices.length > 26) {
      final ema12 = _calculateEMA(widget.prices, 12);
      final ema26 = _calculateEMA(widget.prices, 26);

      _macdLine = List<double>.filled(widget.prices.length, 0);
      for (int i = 0; i < widget.prices.length; i++) {
        _macdLine[i] = ema12[i] - ema26[i];
      }

      _macdSignal = _calculateEMA(_macdLine, 9);
      _macdHistogram = List<double>.filled(widget.prices.length, 0);

      for (int i = 0; i < widget.prices.length; i++) {
        _macdHistogram[i] = _macdLine[i] - _macdSignal[i];
      }
    } else {
      _macdLine = [];
      _macdSignal = [];
      _macdHistogram = [];
    }
  }

  List<double> _calculateEMA(List<double> data, int period) {
    if (data.isEmpty) return [];
    double k = 2 / (period + 1);
    List<double> ema = List<double>.filled(data.length, 0);

    // Initial SMA for seeding EMA 
    double sum = 0;
    for (int i = 0; i < period && i < data.length; i++) {
      sum += data[i];
      ema[i] = sum / (i + 1); // rough seed
    }

    if (data.length >= period) {
      for (int i = period; i < data.length; i++) {
        ema[i] = (data[i] - ema[i - 1]) * k + ema[i - 1];
      }
    }
    return ema;
  }

  @override
  Widget build(BuildContext context) {
    if (widget.prices.isEmpty) {
      return const Center(child: Text('Not enough data points', style: TextStyle(color: Colors.white70)));
    }

    return Column(
      children: [
        Expanded(
          flex: widget.showMACD ? 3 : 1,
          child: _buildMainPriceChart(),
        ),
        if (widget.showMACD && _macdLine.isNotEmpty) ...[
          const SizedBox(height: 16),
          Expanded(
            flex: 1,
            child: _buildMACDSubChart(),
          )
        ]
      ],
    );
  }

  Widget _buildMainPriceChart() {
    final double minY = widget.prices.reduce((a, b) => a < b ? a : b) * 0.95;
    final double maxY = widget.prices.reduce((a, b) => a > b ? a : b) * 1.05;

    return LineChart(
      LineChartData(
        gridData: const FlGridData(show: false),
        titlesData: const FlTitlesData(show: false),
        borderData: FlBorderData(show: false),
        minX: 0,
        maxX: (widget.prices.length - 1).toDouble(),
        minY: minY,
        maxY: maxY,
        lineTouchData: LineTouchData(
          touchTooltipData: LineTouchTooltipData(
            getTooltipColor: (_) => AppColors.surface,
            tooltipRoundedRadius: 12,
            getTooltipItems: (touchedSpots) {
              return touchedSpots.map((spot) {
                return LineTooltipItem(
                  '${widget.dates[spot.x.toInt()]}\n',
                  const TextStyle(color: AppColors.textSecondary, fontSize: 10),
                  children: [
                    TextSpan(
                      text: 'NPR ${spot.y.toStringAsFixed(2)}',
                      style: TextStyle(
                        color: spot.barIndex == 0 ? AppColors.accent : Colors.amber,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                );
              }).toList();
            },
          ),
        ),
        lineBarsData: _buildChartSeries(),
      ),
    );
  }

  Widget _buildMACDSubChart() {
    final double maxMACD = _macdLine.reduce((a, b) => a > b ? a : b).abs() * 1.2;
    final double maxHist = _macdHistogram.isNotEmpty ? _macdHistogram.map((e) => e.abs()).reduce((a, b) => a > b ? a : b) * 1.2 : 0;
    final boundary = maxMACD > maxHist ? maxMACD : maxHist;

    return LineChart(
      LineChartData(
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          horizontalInterval: boundary > 0 ? boundary : null,
          getDrawingHorizontalLine: (value) => value == 0 
            ? FlLine(color: Colors.white24, strokeWidth: 1, dashArray: [5, 5])
            : FlLine(color: Colors.transparent),
        ),
        titlesData: const FlTitlesData(show: false),
        borderData: FlBorderData(show: false),
        minX: 0,
        maxX: (widget.prices.length - 1).toDouble(),
        minY: -boundary,
        maxY: boundary,
        lineTouchData: const LineTouchData(enabled: false), // Synced touches is too complex for basic iteration
        lineBarsData: [
           // Histogram (Rendered as thick line bars conceptually via FlChart hack)
          LineChartBarData(
            spots: _macdHistogram.asMap().entries.map((e) => FlSpot(e.key.toDouble(), e.value)).toList(),
            isCurved: false,
            color: Colors.transparent, // Invisible line
            barWidth: 0,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              color: Colors.cyan.withValues(alpha: 0.3),
              cutOffY: 0,
              applyCutOffY: true,
            ),
            aboveBarData: BarAreaData(
              show: true,
              color: Colors.redAccent.withValues(alpha: 0.3),
              cutOffY: 0,
              applyCutOffY: true,
            ),
          ),
          
          // MACD Line
          LineChartBarData(
            spots: _macdLine.asMap().entries.map((e) => FlSpot(e.key.toDouble(), e.value)).toList(),
            isCurved: true,
            color: Colors.blueAccent,
            barWidth: 2,
            dotData: const FlDotData(show: false),
          ),
          // Signal Line
          LineChartBarData(
            spots: _macdSignal.asMap().entries.map((e) => FlSpot(e.key.toDouble(), e.value)).toList(),
            isCurved: true,
            color: Colors.orangeAccent,
            barWidth: 2,
            dotData: const FlDotData(show: false),
          ),
        ],
      ),
    );
  }

  List<LineChartBarData> _buildChartSeries() {
    final series = <LineChartBarData>[];

    // Main Price Line
    series.add(
      LineChartBarData(
        spots: widget.prices.asMap().entries.map((e) {
          return FlSpot(e.key.toDouble(), e.value);
        }).toList(),
        isCurved: true,
        color: AppColors.accent,
        barWidth: 3,
        isStrokeCapRound: true,
        dotData: const FlDotData(show: false),
        belowBarData: BarAreaData(
          show: true,
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              AppColors.accent.withValues(alpha: 0.2),
              AppColors.accent.withValues(alpha: 0.0),
            ],
          ),
        ),
      ),
    );

    // SMA Overlay
    if (widget.showSMA && _smaData.isNotEmpty) {
      series.add(
        LineChartBarData(
          spots: _smaData.asMap().entries.map((e) {
            return FlSpot(e.key.toDouble(), e.value);
          }).toList(),
          isCurved: true,
          color: Colors.amber,
          barWidth: 2,
          isStrokeCapRound: true,
          dotData: const FlDotData(show: false),
        ),
      );
    }

    return series;
  }
}
