import 'package:freezed_annotation/freezed_annotation.dart';

part 'user_auth.freezed.dart';
part 'user_auth.g.dart';

@freezed
class UserAuth with _$UserAuth {
  const factory UserAuth({
    required String id,
    required String email,
    String? name,
    @Default(false) bool mfaEnabled,
    DateTime? lastLogin,
  }) = _UserAuth;

  factory UserAuth.fromJson(Map<String, dynamic> json) => _$UserAuthFromJson(json);
}

@freezed
class AuthState with _$AuthState {
  const factory AuthState({
    UserAuth? user,
    String? accessToken,
    String? refreshToken,
    @Default(false) bool isLoading,
    String? error,
    @Default(false) bool mfaRequired,
    String? tempUserId,
  }) = _AuthState;
}
