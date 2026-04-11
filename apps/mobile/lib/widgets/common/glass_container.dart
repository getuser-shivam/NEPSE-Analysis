import 'dart:ui';
import 'package:flutter/material.dart';

class GlassContainer extends StatelessWidget {
  final Widget child;
  final double blur;
  final double opacity;
  final BorderRadius? borderRadius;
  final Border? border;
  final EdgeInsetsGeometry? padding;
  final EdgeInsetsGeometry? margin;
  final List<Color>? gradientColors;
  final List<BoxShadow>? boxShadow;

  const GlassContainer({
    super.key,
    required this.child,
    this.blur = 15,
    this.opacity = 0.08,
    this.borderRadius,
    this.border,
    this.padding,
    this.margin,
    this.gradientColors,
    this.boxShadow,
  });

  @override
  Widget build(BuildContext context) {
    final radius = borderRadius ?? BorderRadius.circular(24);
    
    return ClipRRect(
      borderRadius: radius,
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: blur, sigmaY: blur),
        child: Container(
          padding: padding,
          margin: margin,
          decoration: BoxDecoration(
            borderRadius: radius,
            border: border ?? Border.all(color: Colors.white.withValues(alpha: 0.1)),
            boxShadow: boxShadow,
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: gradientColors ?? [
                Colors.white.withValues(alpha: opacity),
                Colors.white.withValues(alpha: opacity * 0.5),
              ],
            ),
          ),
          child: child,
        ),
      ),
    );
  }
}
