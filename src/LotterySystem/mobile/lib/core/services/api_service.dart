import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class ApiService {
  late final Dio _dio;
  
  // Szerver URL - HTTPS domain (megoldja a socket hibát Android 9+-on)
  static const String baseUrl = 'https://liggin.xyz/api';
  
  ApiService() {
    _dio = Dio(BaseOptions(
      baseUrl: baseUrl,
      connectTimeout: const Duration(seconds: 15),
      receiveTimeout: const Duration(seconds: 15),
      headers: {'Content-Type': 'application/json'},
    ));
    
    _dio.interceptors.add(LogInterceptor(
      requestBody: true,
      responseBody: true,
    ));
  }

  void setAuthToken(String token) {
    _dio.options.headers['Authorization'] = 'Bearer $token';
  }

  void clearAuthToken() {
    _dio.options.headers.remove('Authorization');
  }

  // === LOTTERIES ===
  
  /// Elérhető lottó típusok lekérése
  Future<List<dynamic>> getLotteryTypes() async {
    final response = await _dio.get('/admin/lottery-types');
    return response.data['data'];
  }
  
  /// Lottó típusok lekérése (alias a régi metódushoz)
  Future<List<dynamic>> getLotteries() async {
    return getLotteryTypes();
  }

  /// Lottó statisztikák lekérése
  Future<Map<String, dynamic>> getLotteryStats(String lotteryType) async {
    final response = await _dio.get('/admin/stats/$lotteryType');
    return response.data['data'];
  }

  // === PREDICTION ===
  
  /// Predikció generálása
  Future<Map<String, dynamic>> predict(String lotteryType, {int tickets = 4}) async {
    final response = await _dio.get('/admin/predict/$lotteryType', 
      queryParameters: {'tickets': tickets});
    return response.data['data'];
  }
  
  /// Quick generate (alias)
  Future<Map<String, dynamic>> quickGenerate(String lotteryType, {int count = 4}) async {
    return predict(lotteryType, tickets: count);
  }
  
  /// Generate numbers (alias a régi metódushoz)
  Future<Map<String, dynamic>> generateNumbers({
    required String lotteryId,
    required List<String> algorithmIds,
    int count = 1,
  }) async {
    return predict(lotteryId, tickets: count);
  }

  // === DATA MANAGEMENT ===
  
  /// Lottó adatok frissítése
  Future<Map<String, dynamic>> downloadLotteryData(String lotteryType) async {
    final response = await _dio.post('/admin/download/$lotteryType');
    return response.data;
  }
  
  /// Összes lottó frissítése
  Future<Map<String, dynamic>> downloadAllLotteries() async {
    final response = await _dio.post('/admin/download-all');
    return response.data;
  }
  
  /// Adatbázis inicializálása
  Future<Map<String, dynamic>> initDatabase() async {
    final response = await _dio.post('/admin/init-database');
    return response.data;
  }
  
  /// Lottó adatok reset és újratöltés
  Future<Map<String, dynamic>> resetLotteryData(String lotteryType) async {
    final response = await _dio.post('/admin/reset/$lotteryType');
    return response.data;
  }

  // === AUTH (for future use) ===
  
  Future<Map<String, dynamic>> login(String email, String password) async {
    final response = await _dio.post('/auth/login', data: {'email': email, 'password': password});
    return response.data;
  }

  Future<Map<String, dynamic>> register(String name, String email, String password) async {
    final response = await _dio.post('/auth/register', data: {'name': name, 'email': email, 'password': password});
    return response.data;
  }

  Future<void> logout() async {
    await _dio.post('/auth/logout');
    clearAuthToken();
  }

  // === HEALTH ===
  
  Future<Map<String, dynamic>> healthCheck() async {
    final response = await _dio.get('/health'.replaceAll('/api', ''));
    return response.data;
  }
}

final apiServiceProvider = Provider<ApiService>((ref) => ApiService());
