import requests
import json
import os

def test():
    url = 'http://127.0.0.1:5000/predict'
    img_path = r'C:\Users\sarthak24\Downloads\aircraft-surface-damage.v1i.yolov8\test\images\LINE_ALBUM_220324_88_jpg.rf.e89de409b6e3e0c150ee9e8efaf8b421.jpg'
    
    if not os.path.exists(img_path):
        print(f"Error: Test image not found at {img_path}")
        return
        
    print(f"Sending image to {url} for analysis: {os.path.basename(img_path)}...")
    
    with open(img_path, 'rb') as f:
        files = {'image': f}
        try:
            response = requests.post(url, files=files)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print("\n--- Diagnostic Report ---")
                print(f"Success: {data['success']}")
                print(f"Original Image: {data['original_url']}")
                print(f"Annotated Image: {data['annotated_url']}")
                print(f"Caption: {data['caption']}")
                print("\nDetections Counts:")
                for cls, count in data['counts'].items():
                    print(f"  {cls}: {count}")
                print("\nDetailed Damages:")
                for item in data['damages']:
                    print(f"  - Class: {item['class']}, Confidence: {item['confidence']}%, Box: {item['box']}")
                print("\nRecommendations:")
                for rec in data['recommendations']:
                    print(f"  - {rec}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Exception during request: {e}")

if __name__ == '__main__':
    test()
