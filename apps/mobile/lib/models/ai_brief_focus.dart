enum AiBriefFocus { balanced, portfolio, watchlist, risk }

extension AiBriefFocusX on AiBriefFocus {
  String get label => switch (this) {
    AiBriefFocus.balanced => 'Balanced',
    AiBriefFocus.portfolio => 'Portfolio',
    AiBriefFocus.watchlist => 'Watchlist',
    AiBriefFocus.risk => 'Risk',
  };

  String get helperText => switch (this) {
    AiBriefFocus.balanced =>
      'Blend market tone, portfolio posture, and watchlist ideas into one concise brief.',
    AiBriefFocus.portfolio =>
      'Lean harder into holdings, overall return, and where the portfolio needs attention.',
    AiBriefFocus.watchlist =>
      'Prioritize the symbols you are tracking and call out the most actionable names first.',
    AiBriefFocus.risk =>
      'Stress downside, fragility, and the safest next step before any aggressive move.',
  };

  String get promptInstruction => switch (this) {
    AiBriefFocus.balanced =>
      'Keep the response balanced across market tone, the portfolio, and the watchlist.',
    AiBriefFocus.portfolio =>
      'Emphasize portfolio performance, concentration, held positions, and capital preservation.',
    AiBriefFocus.watchlist =>
      'Emphasize watchlist symbols, momentum clues, and what deserves monitoring next.',
    AiBriefFocus.risk =>
      'Emphasize risk controls, fragile spots, and a cautious next action for the investor.',
  };
}
