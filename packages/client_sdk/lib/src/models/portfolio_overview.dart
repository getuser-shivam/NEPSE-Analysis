import 'portfolio_holding.dart';
import 'portfolio_summary.dart';

class PortfolioOverview {
  const PortfolioOverview({required this.holdings, required this.summary});

  final List<PortfolioHolding> holdings;
  final PortfolioSummary summary;

  factory PortfolioOverview.fromJson(Map<String, dynamic> json) {
    final rawHoldings = json['holdings'] as List<dynamic>? ?? const [];

    return PortfolioOverview(
      holdings: rawHoldings
          .map(
            (item) => PortfolioHolding.fromJson(
              Map<String, dynamic>.from(item as Map),
            ),
          )
          .toList(growable: false),
      summary: PortfolioSummary.fromJson(
        Map<String, dynamic>.from(json['summary'] as Map),
      ),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'holdings': holdings.map((holding) => holding.toJson()).toList(),
      'summary': summary.toJson(),
    };
  }
}
