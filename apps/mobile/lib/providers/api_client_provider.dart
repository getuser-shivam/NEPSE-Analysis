import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:nepse_client/nepse_client.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter/material.dart';

// Provides the secure storage singleton
final secureStorageProvider = Provider<FlutterSecureStorage>((ref) {
  return const FlutterSecureStorage();
});

// A global mutable provider that holds an instance of completely authenticated API Client.
// Automatically polls local secure storage to re-hydrate JWT sessions!
final apiClientProvider = Provider<NepseApiClient>((ref) {
  // Uses localhost for development testing, config injection later for prod
  final client = NepseApiClient(baseUrl: 'http://localhost:3000');
  
  // Asynchronously attempt to load existing token from SecureStorage
  WidgetsBinding.instance.addPostFrameCallback((_) async {
    final storage = ref.read(secureStorageProvider);
    final token = await storage.read(key: 'jwt_token');
    if (token != null) {
      client.setAuthToken(token);
    }
  });

  return client;
});
