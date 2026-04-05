class PortfolioSummary {
  const PortfolioSummary({
    required this.totalInvestment,
    required this.totalValue,
    required this.totalGainLoss,
    required this.totalReturnPct,
    required this.numStocks,
  });

  final double totalInvestment;
  final double totalValue;
  final double totalGainLoss;
  final double totalReturnPct;
  final int numStocks;

  factory PortfolioSummary.fromJson(Map<String, dynamic> json) {
    return PortfolioSummary(
      totalInvestment: (json['totalInvestment'] as num).toDouble(),
      totalValue: (json['totalValue'] as num).toDouble(),
      totalGainLoss: (json['totalGainLoss'] as num).toDouble(),
      totalReturnPct: (json['totalReturnPct'] as num).toDouble(),
      numStocks: (json['numStocks'] as num).toInt(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'totalInvestment': totalInvestment,
      'totalValue': totalValue,
      'totalGainLoss': totalGainLoss,
      'totalReturnPct': totalReturnPct,
      'numStocks': numStocks,
    };
  }
}
