import requests
import json

def test_lottery_endpoint():
    """
    Tesztelőkód a lottery végponthoz
    """
    
    url = "http://127.0.0.1:8200/lottery_handler/lottery_handle/get_algorithms_with_scores"
    
    payload = {
        "lottery_type_id": 1,
        "winning_numbers": [4, 14, 26, 29, 50],
        "additional_numbers": [3, 12],
        "test_iterations": 2  # Kevés iteráció a gyors teszteléshez
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("=== LOTTERY ENDPOINT TESZT ===")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("=" * 50)
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print("=" * 50)
        
        if response.status_code == 200:
            print("✅ SIKERES VÁLASZ!")
            response_data = response.json()
            print(json.dumps(response_data, indent=2, ensure_ascii=False))
        else:
            print(f"❌ HIBA: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: A kérés túllépte a 60 másodperces határt")
        
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Nem lehet csatlakozni a szerverhez")
        
    except Exception as e:
        print(f"❌ ÁLTALÁNOS HIBA: {str(e)}")

if __name__ == "__main__":
    test_lottery_endpoint() 