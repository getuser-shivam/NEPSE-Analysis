/// GlassContainer Widget Tests
///
/// Tests for the GlassContainer widget.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:nepse_app/widgets/common/glass_container.dart';

void main() {
  group('GlassContainer Widget', () {
    testWidgets('renders child widget', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: GlassContainer(
              child: const Text('Test Content'),
            ),
          ),
        ),
      );

      expect(find.text('Test Content'), findsOneWidget);
    });

    testWidgets('applies padding when provided', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: GlassContainer(
              padding: const EdgeInsets.all(16),
              child: const Text('Test Content'),
            ),
          ),
        ),
      );

      final container = tester.widget<Container>(
        find.descendant(
          of: find.byType(GlassContainer),
          matching: find.byType(Container),
        ),
      );

      expect(container.padding, equals(const EdgeInsets.all(16)));
    });

    testWidgets('applies margin when provided', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: GlassContainer(
              margin: const EdgeInsets.all(16),
              child: const Text('Test Content'),
            ),
          ),
        ),
      );

      final container = tester.widget<Container>(
        find.descendant(
          of: find.byType(GlassContainer),
          matching: find.byType(Container),
        ),
      );

      expect(container.margin, equals(const EdgeInsets.all(16)));
    });

    testWidgets('applies custom border radius', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: GlassContainer(
              borderRadius: BorderRadius.circular(8),
              child: const Text('Test Content'),
            ),
          ),
        ),
      );

      final clipRRect = tester.widget<ClipRRect>(
        find.byType(ClipRRect),
      );

      expect(clipRRect.borderRadius, equals(BorderRadius.circular(8)));
    });

    testWidgets('uses default border radius when not provided',
        (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: GlassContainer(
              child: const Text('Test Content'),
            ),
          ),
        ),
      );

      final clipRRect = tester.widget<ClipRRect>(
        find.byType(ClipRRect),
      );

      expect(clipRRect.borderRadius, equals(BorderRadius.circular(24)));
    });

    testWidgets('applies custom border', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: GlassContainer(
              border: Border.all(color: Colors.red, width: 2),
              child: const Text('Test Content'),
            ),
          ),
        ),
      );

      final container = tester.widget<Container>(
        find.descendant(
          of: find.byType(GlassContainer),
          matching: find.byType(Container),
        ),
      );

      final decoration = container.decoration as BoxDecoration;
      expect(decoration.border, equals(Border.all(color: Colors.red, width: 2)));
    });

    testWidgets('applies custom blur value', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: GlassContainer(
              blur: 20,
              child: const Text('Test Content'),
            ),
          ),
        ),
      );

      final backdropFilter = tester.widget<BackdropFilter>(
        find.byType(BackdropFilter),
      );

      expect(backdropFilter.filter.sigmaX, equals(20));
      expect(backdropFilter.filter.sigmaY, equals(20));
    });

    testWidgets('applies custom opacity', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: GlassContainer(
              opacity: 0.5,
              child: const Text('Test Content'),
            ),
          ),
        ),
      );

      final container = tester.widget<Container>(
        find.descendant(
          of: find.byType(GlassContainer),
          matching: find.byType(Container),
        ),
      );

      final decoration = container.decoration as BoxDecoration;
      final gradient = decoration.gradient as LinearGradient;
      
      expect(gradient.colors.first.withValues(alpha: 1).withValues(alpha: 1), equals(Colors.white.withValues(alpha: 0.5)));
    });

    testWidgets('applies custom gradient colors', (WidgetTester tester) async {
      final customColors = [Colors.blue, Colors.purple];

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: GlassContainer(
              gradientColors: customColors,
              child: const Text('Test Content'),
            ),
          ),
        ),
      );

      final container = tester.widget<Container>(
        find.descendant(
          of: find.byType(GlassContainer),
          matching: find.byType(Container),
        ),
      );

      final decoration = container.decoration as BoxDecoration;
      final gradient = decoration.gradient as LinearGradient;
      
      expect(gradient.colors, equals(customColors));
    });

    testWidgets('applies custom box shadow', (WidgetTester tester) async {
      final boxShadow = [
        BoxShadow(
          color: Colors.black.withValues(alpha: 0.2),
          blurRadius: 10,
          offset: const Offset(0, 4),
        ),
      ];

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: GlassContainer(
              boxShadow: boxShadow,
              child: const Text('Test Content'),
            ),
          ),
        ),
      );

      final container = tester.widget<Container>(
        find.descendant(
          of: find.byType(GlassContainer),
          matching: find.byType(Container),
        ),
      );

      final decoration = container.decoration as BoxDecoration;
      expect(decoration.boxShadow, equals(boxShadow));
    });
  });
}
