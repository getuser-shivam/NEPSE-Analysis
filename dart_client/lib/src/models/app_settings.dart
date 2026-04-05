class AppSettings {
  const AppSettings({
    required this.id,
    required this.name,
    required this.autoSaveInterval,
    required this.maxDataAgeDays,
    required this.backupEnabled,
    required this.chartStyle,
    required this.defaultPeriod,
    required this.maxWatchlistSize,
    required this.logLevel,
    required this.refreshInterval,
    required this.createdAt,
    required this.updatedAt,
  });

  final String id;
  final String name;
  final int autoSaveInterval;
  final int maxDataAgeDays;
  final bool backupEnabled;
  final String chartStyle;
  final String defaultPeriod;
  final int maxWatchlistSize;
  final String logLevel;
  final int refreshInterval;
  final DateTime createdAt;
  final DateTime updatedAt;

  factory AppSettings.fromJson(Map<String, dynamic> json) {
    return AppSettings(
      id: json['id'] as String,
      name: json['name'] as String,
      autoSaveInterval: (json['autoSaveInterval'] as num).toInt(),
      maxDataAgeDays: (json['maxDataAgeDays'] as num).toInt(),
      backupEnabled: json['backupEnabled'] as bool,
      chartStyle: json['chartStyle'] as String,
      defaultPeriod: json['defaultPeriod'] as String,
      maxWatchlistSize: (json['maxWatchlistSize'] as num).toInt(),
      logLevel: json['logLevel'] as String,
      refreshInterval: (json['refreshInterval'] as num).toInt(),
      createdAt: DateTime.parse(json['createdAt'] as String),
      updatedAt: DateTime.parse(json['updatedAt'] as String),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'autoSaveInterval': autoSaveInterval,
      'maxDataAgeDays': maxDataAgeDays,
      'backupEnabled': backupEnabled,
      'chartStyle': chartStyle,
      'defaultPeriod': defaultPeriod,
      'maxWatchlistSize': maxWatchlistSize,
      'logLevel': logLevel,
      'refreshInterval': refreshInterval,
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt.toIso8601String(),
    };
  }
}
