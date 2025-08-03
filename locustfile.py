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
        
        try:
            for i, color in enumerate(colors):
                # Create a simple colored image
                img_array = np.full((224, 224, 3), color, dtype=np.uint8)
                
                # Add some noise to make it more realistic
                noise = np.random.randint(-25, 26, (224, 224, 3), dtype=np.int16)
                img_array = img_array.astype(np.int16) + noise
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                
                # Convert to PIL Image with explicit mode
                img = Image.fromarray(img_array, mode='RGB')
                
                # Convert to bytes - FIX: Don't seek(0) on BytesIO after getvalue()
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=90)
                img_data = img_buffer.getvalue()  # Get the data
                img_buffer.close()  # Clean up
                
                self.test_images.append({
                    'name': f'test_image_{i}.jpg',
                    'data': img_data,
                    'class': ['cats', 'dogs', 'birds'][i]
                })
                
            print(f"Created {len(self.test_images)} test images successfully")
            
        except Exception as e:
            print(f"Error creating test images: {e}")
            # Create fallback simple images if numpy fails
            self.create_fallback_images()
    
    def create_fallback_images(self):
        """Create simple fallback images if numpy creation fails"""
        self.test_images = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB colors
        class_names = ['cats', 'dogs', 'birds']
        
        for i, (color, class_name) in enumerate(zip(colors, class_names)):
            # Create image directly with PIL
            img = Image.new('RGB', (224, 224), color)
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=90)
            img_data = img_buffer.getvalue()
            img_buffer.close()
            
            self.test_images.append({
                'name': f'fallback_image_{i}.jpg',
                'data': img_data,
                'class': class_name
            })
    
    @task(10)
    def predict_image(self):
        """Test image prediction endpoint"""
        if not hasattr(self, 'test_images') or not self.test_images:
            self.create_test_images()
            
        # Select a random test image
        test_image = random.choice(self.test_images)
        
        # Prepare the file for upload
        files = {
            'file': (test_image['name'], test_image['data'], 'image/jpeg')
        }
        
        # Make prediction request
        with self.client.post("/predict", files=files, catch_response=True, timeout=30) as response:
            try:
                if response.status_code == 200:
                    result = response.json()
                    if 'predicted_class' in result:
                        response.success()
                    else:
                        response.failure("Missing predicted_class in response")
                elif response.status_code == 413:
                    response.failure("File too large")
                elif response.status_code == 400:
                    response.failure("Bad request - check file format")
                else:
                    response.failure(f"Status code: {response.status_code}")
            except Exception as e:
                response.failure(f"Request failed: {str(e)}")
    
    @task(2)
    def get_status(self):
        """Test status endpoint"""
        with self.client.get("/status", catch_response=True, timeout=10) as response:
            try:
                if response.status_code == 200:
                    result = response.json()
                    if 'system' in result and 'model' in result:
                        response.success()
                    else:
                        response.failure("Invalid status response format")
                else:
                    response.failure(f"Status code: {response.status_code}")
            except Exception as e:
                response.failure(f"Request failed: {str(e)}")
    
    @task(1)
    def get_dashboard(self):
        """Test dashboard endpoint"""
        with self.client.get("/", catch_response=True, timeout=10) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_visualizations(self):
        """Test visualizations endpoint"""
        with self.client.get("/visualizations", catch_response=True, timeout=10) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_model_summary(self):
        """Test model summary endpoint"""
        with self.client.get("/model_summary", catch_response=True, timeout=10) as response:
            try:
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
            except Exception as e:
                response.failure(f"Request failed: {str(e)}")

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

# Additional user class for mixed load testing
class MixedLoadUser(MLPipelineUser):
    """User class that mixes different behaviors"""
    wait_time = between(0.5, 2)
    
    @task(15)
    def mixed_predict(self):
        """Mixed prediction behavior"""
        self.predict_image()
    
    @task(3)
    def mixed_status_check(self):
        """More frequent status checks"""
        self.get_status()

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
    
    @staticmethod
    def endurance_test():
        """Endurance test: moderate load for extended period"""
        return {
            'users': 100,
            'spawn_rate': 5,
            'run_time': '30m'
        }

# Performance monitoring helper
class PerformanceMonitor:
    """Helper class to track performance metrics"""
    
    def __init__(self):
        self.response_times = []
        self.error_count = 0
        self.success_count = 0
    
    def record_response(self, response_time, success=True):
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def get_stats(self):
        if not self.response_times:
            return {}
        
        return {
            'avg_response_time': sum(self.response_times) / len(self.response_times),
            'min_response_time': min(self.response_times),
            'max_response_time': max(self.response_times),
            'success_rate': self.success_count / (self.success_count + self.error_count) * 100,
            'total_requests': len(self.response_times)
        }

# Example usage commands:
"""
# Normal load test:
locust -f locustfile.py --host=http://localhost:8000 -u 50 -r 2 -t 5m --html=report.html

# Stress test:
locust -f locustfile.py --host=http://localhost:8000 -u 200 -r 10 -t 10m --csv=stress_test

# Spike test:
locust -f locustfile.py --host=http://localhost:8000 -u 500 -r 50 -t 3m

# Mixed user types test:
locust -f locustfile.py --host=http://localhost:8000 -u 100 -r 5 -t 15m

# Endurance test:
locust -f locustfile.py --host=http://localhost:8000 -u 100 -r 5 -t 30m

# With specific user class:
locust -f locustfile.py --host=http://localhost:8000 MLPipelineUser -u 50 -r 2 -t 10m

# Multiple user types (weights automatically balanced):
locust -f locustfile.py --host=http://localhost:8000 -u 150 -r 10 -t 10m
"""