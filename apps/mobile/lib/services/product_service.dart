import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/product.dart';
import 'api_settings_service.dart';

final productServiceProvider = Provider((ref) => ProductService(ref));

class ProductService {
  final Ref _ref;
  final ApiSettingsService _apiSettingsService = ApiSettingsService();

  ProductService(this._ref);

  Future<List<Product>> getProducts({
    String? category,
    String? search,
    int page = 1,
    int limit = 20,
  }) async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();

    final queryParams = {
      if (category != null) 'category': category,
      if (search != null) 'search': search,
      'page': page.toString(),
      'limit': limit.toString(),
    };

    final uri = Uri.parse(
      '$baseUrl/api/products',
    ).replace(queryParameters: queryParams);

    // In a real app, we'd add the auth token here from a secure storage provider
    final response = await http.get(uri);

    if (response.statusCode == 200) {
      final body = json.decode(response.body);
      final List<dynamic> data = body['data'];
      return data.map((json) => Product.fromJson(json)).toList();
    } else {
      throw Exception('Failed to load products: ${response.statusCode}');
    }
  }

  Future<Product> getProductDetails(String id) async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();
    final uri = Uri.parse('$baseUrl/api/products/$id');

    final response = await http.get(uri);

    if (response.statusCode == 200) {
      return Product.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to load product details: ${response.statusCode}');
    }
  }
}
