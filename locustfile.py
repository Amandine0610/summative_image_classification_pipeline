from locust import HttpUser, task, between
import io
import base64
from PIL import Image
import numpy as np
import random
import time

class MLPipelineUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Called when a user starts"""
        self.create_test_images()
    
    def create_test_images(self):
        """Create sample test images for load testing"""
        self.test_images = []
        
        # Create different types of test images
        colors = [
            (255, 100, 100),  # Red (cats)
            (100, 255, 100),  # Green (dogs)
            (100, 100, 255),  # Blue (birds)
        ]
        
        for i, color in enumerate(colors):
            # Create a simple colored image
            img_array = np.full((224, 224, 3), color, dtype=np.uint8)
            
            # Add some noise
            noise = np.random.randint(0, 50, (224, 224, 3))
            img_array = np.clip(img_array + noise, 0, 255)
            
            # Convert to PIL Image
            img = Image.fromarray(img_array)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            self.test_images.append({
                'name': f'test_image_{i}.jpg',
                'data': img_bytes.getvalue(),
                'class': ['cats', 'dogs', 'birds'][i]
            })
    
    @task(10)
    def predict_image(self):
        """Test image prediction endpoint"""
        # Select a random test image
        test_image = random.choice(self.test_images)
        
        # Prepare the file for upload
        files = {
            'file': (test_image['name'], test_image['data'], 'image/jpeg')
        }
        
        # Make prediction request
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if 'predicted_class' in result:
                    response.success()
                else:
                    response.failure("Missing predicted_class in response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def get_status(self):
        """Test status endpoint"""
        with self.client.get("/status", catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if 'system' in result and 'model' in result:
                    response.success()
                else:
                    response.failure("Invalid status response format")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_dashboard(self):
        """Test dashboard endpoint"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_visualizations(self):
        """Test visualizations endpoint"""
        with self.client.get("/visualizations", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_model_summary(self):
        """Test model summary endpoint"""
        with self.client.get("/model_summary", catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if 'summary' in result:
                    response.success()
                else:
                    response.failure("Missing summary in response")
            elif response.status_code == 400:
                # Model not loaded is acceptable
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

class HighLoadUser(MLPipelineUser):
    """User class for high-load testing"""
    wait_time = between(0.1, 0.5)  # Much shorter wait time
    
    @task(20)
    def rapid_predictions(self):
        """Rapid fire predictions for load testing"""
        self.predict_image()

class BurstLoadUser(MLPipelineUser):
    """User class for burst load testing"""
    wait_time = between(0, 0.1)  # Very short wait time
    
    @task(30)
    def burst_predictions(self):
        """Burst predictions for stress testing"""
        self.predict_image()

# Custom load test scenarios
class LoadTestScenarios:
    """Different load testing scenarios"""
    
    @staticmethod
    def normal_load():
        """Normal load: 10-50 users, gradual ramp-up"""
        return {
            'users': 50,
            'spawn_rate': 2,
            'run_time': '5m'
        }
    
    @staticmethod
    def stress_test():
        """Stress test: 100-200 users, quick ramp-up"""
        return {
            'users': 200,
            'spawn_rate': 10,
            'run_time': '10m'
        }
    
    @staticmethod
    def spike_test():
        """Spike test: sudden load increase"""
        return {
            'users': 500,
            'spawn_rate': 50,
            'run_time': '3m'
        }

# Example usage commands:
# Normal load test:
# locust -f locustfile.py --host=http://localhost:8000 -u 50 -r 2 -t 5m

# Stress test:
# locust -f locustfile.py --host=http://localhost:8000 -u 200 -r 10 -t 10m

# Spike test:
# locust -f locustfile.py --host=http://localhost:8000 -u 500 -r 50 -t 3m

# Multiple containers test:
# locust -f locustfile.py --host=http://localhost:80 -u 100 -r 5 -t 10m