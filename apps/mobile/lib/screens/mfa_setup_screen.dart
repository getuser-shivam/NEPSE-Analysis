import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/app_providers.dart';
import '../theme/app_theme.dart';
import '../widgets/common/glass_container.dart';

class MfaSetupScreen extends ConsumerStatefulWidget {
  const MfaSetupScreen({super.key});

  @override
  ConsumerState<MfaSetupScreen> createState() => _MfaSetupScreenState();
}

class _MfaSetupScreenState extends ConsumerState<MfaSetupScreen> {
  Map<String, dynamic>? _setupData;
  bool _isLoading = false;

  Future<void> _startSetup() async {
    setState(() => _isLoading = true);
    // API call to POST /api/mfa/setup
    // For this walkthrough, we'll assume the provider handles it or we call service directly
    setState(() => _isLoading = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('MFA Setup')),
      body: DecoratedBox(
        decoration: const BoxDecoration(gradient: AppTheme.backgroundGradient),
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: GlassContainer(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.qr_code_2_outlined, size: 80, color: AppColors.primary),
                const SizedBox(height: 24),
                const Text(
                  'Enable Multi-Factor Authentication',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 16),
                const Text(
                  'Add an extra layer of security to your account. Use an authenticator app (Google Authenticator, Authy) to scan the code.',
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 32),
                if (_setupData == null)
                  ElevatedButton(
                    onPressed: _isLoading ? null : _startSetup,
                    child: const Text('Generate Secret Key'),
                  )
                else ...[
                  SelectableText(
                    'Secret Key: ${_setupData!['secret']}',
                    style: const TextStyle(fontWeight: FontWeight.bold, color: AppColors.accent),
                  ),
                  const SizedBox(height: 16),
                  const Text('Enter the code from your app below to verify.'),
                  const SizedBox(height: 16),
                  TextField(
                    decoration: InputDecoration(
                      labelText: '6-digit code',
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}
