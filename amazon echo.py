amazon echo 

#!/usr/bin/env python3
import time
import os
import subprocess
import random
import math
from datetime import datetime

class AmazonEchoShowHPC:
    def __init__(self, samples=20000, duration=600):
        self.samples = samples
        self.duration = duration
        self.interval = duration / samples
        self.data = []
        
        # Amazon Echo Show specific workload states
        self.workload_states = {
            'idle_homescreen': 0,
            'alexa_wakeword': 1,
            'voice_processing': 2,
            'video_call_processing': 3,
            'media_streaming': 4,
            'smart_home_control': 5,
            'weather_display': 6,
            'news_briefing': 7,
            'recipe_display': 8,
            'shopping_list': 9,
            'calendar_display': 10,
            'video_playback': 11
        }
        self.current_state = 'idle_homescreen'
        self.state_timer = 0
        
        # Echo Show-specific features
        self.video_call_active = False
        self.alexa_responding = False
        self.media_playing = False
        self.smart_home_devices = 15
        self.weather_alerts = 2
        
        # Video call tracking
        self.video_call_duration = 0
        self.video_call_start_time = 0
        
        # Hardware performance counters
        self.hw_events = [
            'cpu-cycles',
            'instructions',
            'branch-instructions',
            'branch-misses',
            'cache-references',
            'cache-misses',
            'L1-dcache-loads',
            'L1-dcache-load-misses',
            'LLC-loads',
            'LLC-load-misses',
            'stalled-cycles-frontend',
            'stalled-cycles-backend',
            'bus-cycles'
        ]
        
        self.working_events = self._test_hardware_events()
        print(f"Available hardware counters: {len(self.working_events)}")

    def _sigmoid(self, x):
        """Sigmoid function implementation for older Python versions"""
        return 1 / (1 + math.exp(-x))

    def _test_hardware_events(self):
        """Test which hardware performance counters work"""
        working = []
        for event in self.hw_events:
            try:
                result = subprocess.run(
                    ['perf', 'stat', '-e', event, 'sleep', '0.001'],
                    capture_output=True, text=True, timeout=2
                )
                for line in result.stderr.split('\n'):
                    if event in line and line.strip() and line[0].isdigit():
                        working.append(event)
                        break
            except:
                continue
        return working

    def simulate_echo_show_workload(self):
        """Simulate Amazon Echo Show specific workload patterns"""
        self.state_timer += 1
        workload = 0
        
        # Track video call duration
        if self.video_call_active:
            self.video_call_duration += 1
        
        # Echo Show-specific state transitions with realistic probabilities
        if self.current_state == 'idle_homescreen':
            # Background home screen with rotating content
            workload = self._simulate_homescreen_rotating_content()
            
            # Frequent activations (10% chance) - Echo Show is often used
            if random.random() < 0.10 or self.state_timer > 35:
                next_states = ['alexa_wakeword', 'weather_display', 'news_briefing', 'smart_home_control', 'video_call_processing']
                weights = [0.4, 0.15, 0.1, 0.15, 0.2]  # Wake word and video calls more common
                self.current_state = random.choices(next_states, weights=weights)[0]
                self.state_timer = 0
                
                # If starting video call, set active flag
                if self.current_state == 'video_call_processing':
                    self.video_call_active = True
                    self.video_call_start_time = time.time()
        
        elif self.current_state == 'alexa_wakeword':
            workload = self._simulate_alexa_wakeword_detection()
            if random.random() < 0.30 or self.state_timer > 12:
                # Higher probability to transition to video calls
                next_states = ['voice_processing', 'video_call_processing', 'weather_display', 'media_streaming']
                weights = [0.5, 0.3, 0.1, 0.1]
                self.current_state = random.choices(next_states, weights=weights)[0]
                self.state_timer = 0
                
                if self.current_state == 'video_call_processing':
                    self.video_call_active = True
                    self.video_call_start_time = time.time()
        
        elif self.current_state == 'voice_processing':
            workload = self._simulate_voice_processing()
            self.alexa_responding = True
            if random.random() < 0.40 or self.state_timer > 18:
                # Common Echo Show responses - include video calls
                next_states = ['weather_display', 'news_briefing', 'recipe_display', 'shopping_list', 'calendar_display', 'media_streaming', 'video_call_processing']
                weights = [0.15, 0.15, 0.1, 0.1, 0.1, 0.2, 0.2]
                self.current_state = random.choices(next_states, weights=weights)[0]
                self.alexa_responding = False
                self.state_timer = 0
                
                if self.current_state == 'video_call_processing':
                    self.video_call_active = True
                    self.video_call_start_time = time.time()
        
        elif self.current_state == 'video_call_processing':
            workload = self._simulate_video_call_processing()
            self.video_call_active = True
            
            # Video calls typically last longer - 2-10 minutes
            call_duration_threshold = random.randint(120, 600)  # 2-10 minutes in samples
            if random.random() < 0.08 or self.video_call_duration > call_duration_threshold:
                self.current_state = 'idle_homescreen'
                self.video_call_active = False
                self.video_call_duration = 0
                self.state_timer = 0
        
        elif self.current_state == 'media_streaming':
            workload = self._simulate_media_streaming()
            self.media_playing = True
            if random.random() < 0.18 or self.state_timer > 120:
                if random.random() < 0.4:
                    self.current_state = 'video_playback'
                else:
                    self.current_state = 'idle_homescreen'
                self.media_playing = False
                self.state_timer = 0
        
        elif self.current_state == 'smart_home_control':
            workload = self._simulate_smart_home_control()
            if random.random() < 0.25 or self.state_timer > 25:
                self.current_state = 'idle_homescreen'
                self.state_timer = 0
        
        elif self.current_state == 'weather_display':
            workload = self._simulate_weather_display()
            if random.random() < 0.22 or self.state_timer > 40:
                self.current_state = 'idle_homescreen'
                self.state_timer = 0
        
        elif self.current_state == 'news_briefing':
            workload = self._simulate_news_briefing()
            if random.random() < 0.20 or self.state_timer > 45:
                self.current_state = 'idle_homescreen'
                self.state_timer = 0
        
        elif self.current_state == 'recipe_display':
            workload = self._simulate_recipe_display()
            if random.random() < 0.15 or self.state_timer > 90:
                self.current_state = 'idle_homescreen'
                self.state_timer = 0
        
        elif self.current_state == 'shopping_list':
            workload = self._simulate_shopping_list()
            if random.random() < 0.28 or self.state_timer > 30:
                self.current_state = 'idle_homescreen'
                self.state_timer = 0
        
        elif self.current_state == 'calendar_display':
            workload = self._simulate_calendar_display()
            if random.random() < 0.24 or self.state_timer > 35:
                self.current_state = 'idle_homescreen'
                self.state_timer = 0
        
        elif self.current_state == 'video_playback':
            workload = self._simulate_video_playback()
            if random.random() < 0.14 or self.state_timer > 150:
                self.current_state = 'idle_homescreen'
                self.state_timer = 0
        
        # Add Amazon-specific micro-interactions (35% of samples)
        if random.random() < 0.35 and self.current_state != 'idle_homescreen':
            workload += self._generate_echo_micro_interaction()
        
        # Ensure minimum workload in all active states
        if workload < 40 and self.current_state != 'idle_homescreen':
            workload += self._generate_minimum_echo_workload()
        
        workload_intensity = min(100, workload / 35)  # Scale to 0-100
        
        return workload, workload_intensity

    def _generate_minimum_echo_workload(self):
        """Generate minimum guaranteed Echo Show workload"""
        workload = 0
        for i in range(35):
            workload += i * (random.random() + 0.6)
        return workload

    def _generate_echo_micro_interaction(self):
        """Generate Amazon-specific micro-interactions"""
        workload = 0
        interaction_intensity = random.randint(50, 200)
        for i in range(interaction_intensity // 8):
            workload += math.sin(i * 0.12) * 20 + math.cos(i * 0.08) * 15
        return workload

    def _simulate_homescreen_rotating_content(self):
        """Simulate Echo Show home screen with rotating widgets"""
        workload = 15  # Base background workload
        # Rotating content widgets (weather, news, photos, etc.)
        for i in range(70):
            workload += math.cos(i * 0.06) * 8
            workload += (i % 20) * 1.2  # Widget rotation cycles
            
            # Photo frame animations
            if i % 25 == 0:
                workload += 25  # Photo transition
        
        # Clock and weather updates
        for i in range(40):
            workload += (i % 10) * 2
        
        return workload + random.randint(10, 35)

    def _simulate_alexa_wakeword_detection(self):
        """Simulate Alexa wake word detection with far-field microphones"""
        workload = 55
        # Advanced audio processing for "Alexa" detection
        for i in range(160):
            # Audio signal processing
            real = math.cos(2 * math.pi * i / 120) * 100
            imag = math.sin(2 * math.pi * i / 120) * 100
            workload += abs(complex(real, imag)) * 0.12
            
            # Neural network inference for wake word
            if i % 40 == 0:
                workload += self._sigmoid(i * 0.02) * 45
        
        return workload + random.randint(25, 80)

    def _simulate_voice_processing(self):
        """Simulate Alexa voice command processing"""
        workload = 85
        # Natural Language Understanding (NLU)
        alexa_commands = [
            "what's the weather today",
            "play some jazz music",
            "show me my front door camera",
            "add milk to shopping list",
            "what's on my calendar",
            "set a timer for 10 minutes",
            "turn off living room lights",
            "show me recipes for chicken",
            "video call mom",  # Video call command
            "call dad on echo show",  # Video call command
            "read me the news"
        ]
        command = random.choice(alexa_commands)
        
        # Voice-to-text processing
        for i, char in enumerate(command):
            workload += ord(char) * (i % 10) * 0.7
        
        # Intent recognition and entity extraction
        for i in range(150):
            workload += math.tanh(i * 0.03) * 40 + math.sinh(i * 0.015) * 25
        
        # Cloud service communication simulation
        for i in range(80):
            workload += (i * 5) % 120 * 0.4
        
        return workload + random.randint(40, 110)

    def _simulate_video_call_processing(self):
        """Simulate Amazon video call processing"""
        workload = 145  # Higher workload for video calls
        # Video encoding/decoding for calls
        for i in range(320):
            workload += (i * 18) % 420  # Video compression
            workload += (i * 11) % 380   # Audio processing
            workload += math.sin(i * 0.09) * 75  # Real-time encoding
            
            # Face detection and tracking
            if i % 45 == 0:
                workload += 55
        
        # Echo Show camera processing
        for i in range(150):
            workload += (i % 35) * 4.2
            
            # Video stabilization
            if i % 25 == 0:
                workload += 35
        
        # Network bandwidth simulation for video streaming
        for i in range(100):
            workload += math.cos(i * 0.05) * 25
        
        return workload + random.randint(80, 180)

    def _simulate_media_streaming(self):
        """Simulate Amazon Music/Prime Music streaming"""
        workload = 65
        # Audio streaming and processing
        for i in range(200):
            workload += math.sin(i * 0.07) * 35
            workload += (i * 8) % 250 * 0.5
        
        # Album art and UI rendering
        for i in range(100):
            workload += (i % 25) * 2.8
        
        return workload + random.randint(30, 90)

    def _simulate_smart_home_control(self):
        """Simulate smart home device control"""
        workload = 60
        # Device communication and control
        devices = ['lights', 'thermostat', 'cameras', 'locks', 'plugs']
        for device_idx, device in enumerate(devices):
            # Device state management
            for i in range(80):
                workload += math.sin(device_idx * 0.8 + i * 0.05) * 18
                workload += (device_idx * 20 + i * 3) % 150 * 0.4
            
            # Cloud synchronization
            for i in range(60):
                workload += (i % 15) * 2.5
        
        return workload + random.randint(25, 75)

    def _simulate_weather_display(self):
        """Simulate weather information display with animations"""
        workload = 50
        # Weather data processing and visualization
        for i in range(150):
            workload += math.cos(i * 0.05) * 22
            workload += (i % 35) * 1.8
            
            # Weather animation (rain, sun, clouds)
            if i % 40 == 0:
                workload += 35
        
        # Forecast calculations
        for i in range(80):
            workload += (i * 2) % 100 * 0.6
        
        return workload + random.randint(20, 60)

    def _simulate_news_briefing(self):
        """Simulate news briefing with text-to-speech"""
        workload = 70
        # News content processing
        for i in range(180):
            workload += math.sin(i * 0.06) * 28
            workload += (i * 6) % 200 * 0.5
        
        # Text-to-speech processing
        for i in range(120):
            workload += (i % 20) * 3.2
            
            # Speech synthesis bursts
            if i % 25 == 0:
                workload += 40
        
        return workload + random.randint(35, 85)

    def _simulate_recipe_display(self):
        """Simulate recipe display with step-by-step instructions"""
        workload = 75
        # Recipe content rendering
        for i in range(220):
            workload += math.cos(i * 0.04) * 30
            workload += (i % 45) * 1.6
        
        # Step-by-step navigation
        for i in range(100):
            workload += (i * 3) % 120 * 0.7
        
        # Cooking timer management
        for i in range(60):
            workload += (i % 12) * 2.5
        
        return workload + random.randint(40, 95)

    def _simulate_shopping_list(self):
        """Simulate shopping list management"""
        workload = 45
        # List management operations
        for i in range(120):
            workload += math.sin(i * 0.08) * 20
            workload += (i % 28) * 1.4
        
        # Item categorization and sorting
        for i in range(80):
            workload += (i * 2) % 90 * 0.5
        
        return workload + random.randint(20, 55)

    def _simulate_calendar_display(self):
        """Simulate calendar and reminder display"""
        workload = 55
        # Calendar rendering and event processing
        for i in range(160):
            workload += math.cos(i * 0.07) * 25
            workload += (i % 32) * 1.7
        
        # Event notification system
        for i in range(90):
            workload += (i % 18) * 2.8
        
        return workload + random.randint(25, 65)

    def _simulate_video_playback(self):
        """Simulate Prime Video playback"""
        workload = 95
        # Video decoding and rendering
        for i in range(250):
            workload += (i * 12) % 320  # Video decoding
            workload += (i * 7) % 280   # Audio synchronization
            workload += math.tan(i * 0.02) * 35  # Rendering pipeline
        
        # Streaming buffer management
        for i in range(120):
            workload += (i % 25) * 3.5
        
        return workload + random.randint(50, 120)

    def read_hardware_performance_counters(self):
        """Read hardware performance counters"""
        if not self.working_events:
            return self._generate_realistic_echo_hw_data()
        
        try:
            events_batch = self.working_events[:6]
            event_string = ','.join(events_batch)
            
            cmd = ['perf', 'stat', '-e', event_string, 'sleep', '0.001']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            
            counters = {}
            for line in result.stderr.split('\n'):
                for event in events_batch:
                    if event in line and line.strip() and line[0].isdigit():
                        parts = line.split()
                        value = parts[0].replace(',', '')
                        if value.isdigit():
                            counters[event] = int(value)
            return counters
            
        except:
            return self._generate_realistic_echo_hw_data()

    def _generate_realistic_echo_hw_data(self):
        """Generate realistic Echo Show hardware counter data"""
        base = time.time_ns() % 1000000
        # Increase counters when video call is active
        multiplier = 1.3 if self.video_call_active else 1.0
        
        return {
            'cpu-cycles': int((2000000 + (base * 35) % 800000) * multiplier),
            'instructions': int((1800000 + (base * 27) % 700000) * multiplier),
            'branch-instructions': int((120000 + (base * 15) % 60000) * multiplier),
            'branch-misses': int((4000 + (base * 11) % 2500) * multiplier),
            'cache-references': int((180000 + (base * 21) % 90000) * multiplier),
            'cache-misses': int((9000 + (base * 17) % 6000) * multiplier),
            'L1-dcache-loads': int((150000 + (base * 29) % 80000) * multiplier),
            'L1-dcache-load-misses': int((7000 + (base * 9) % 4500) * multiplier),
            'LLC-loads': int((35000 + (base * 5) % 22000) * multiplier),
            'LLC-load-misses': int((2000 + (base * 3) % 1200) * multiplier),
            'stalled-cycles-frontend': int((160000 + (base * 39) % 110000) * multiplier),
            'stalled-cycles-backend': int((140000 + (base * 41) % 90000) * multiplier),
            'bus-cycles': int((60000 + (base * 15) % 30000) * multiplier)
        }

    def calculate_hardware_metrics(self, raw_counters):
        """Calculate hardware performance metrics"""
        cycles = max(1, raw_counters.get('cpu-cycles', 1))
        instructions = raw_counters.get('instructions', 1600000)
        branches = max(1, raw_counters.get('branch-instructions', 90000))
        branch_misses = raw_counters.get('branch-misses', 3000)
        cache_refs = max(1, raw_counters.get('cache-references', 150000))
        cache_misses = raw_counters.get('cache-misses', 8000)
        l1_loads = max(1, raw_counters.get('L1-dcache-loads', 130000))
        l1_misses = raw_counters.get('L1-dcache-load-misses', 6000)
        llc_loads = max(1, raw_counters.get('LLC-loads', 30000))
        llc_misses = raw_counters.get('LLC-load-misses', 1600)
        frontend_stalls = raw_counters.get('stalled-cycles-frontend', 130000)
        backend_stalls = raw_counters.get('stalled-cycles-backend', 110000)
        
        return {
            # CPU Pipeline (5 metrics)
            'hw_cpu_cycles': cycles,
            'hw_instructions_retired': instructions,
            'hw_instructions_per_cycle': instructions / cycles,
            'hw_branch_instructions': branches,
            'hw_branch_miss_rate': branch_misses / branches,
            
            # Cache Hierarchy (5 metrics)
            'hw_cache_references': cache_refs,
            'hw_cache_miss_rate': cache_misses / cache_refs,
            'hw_l1d_cache_access': l1_loads,
            'hw_l1d_cache_miss_rate': l1_misses / l1_loads,
            'hw_llc_cache_access': llc_loads,
            
            # Memory & Efficiency (5 metrics)
            'hw_llc_cache_miss_rate': llc_misses / llc_loads,
            'hw_frontend_stall_ratio': frontend_stalls / cycles,
            'hw_backend_stall_ratio': backend_stalls / cycles,
            'hw_memory_bandwidth_mbps': (cache_misses * 64) / (1024 * 1024),
            'hw_cache_efficiency': max(0, 100 - (cache_misses / cache_refs) * 100)
        }

    def get_system_context(self):
        """Get system context"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = float(f.read().strip()) / 1000.0
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
                cpu_freq = float(f.read().strip()) / 1000.0
            return {'cpu_temp': cpu_temp, 'cpu_freq': cpu_freq}
        except:
            return {'cpu_temp': 48.0, 'cpu_freq': 1200.0}

    def collect_sample(self):
        """Collect one sample"""
        timestamp = datetime.now()
        
        # Simulate Echo Show workload
        workload_raw, workload_intensity = self.simulate_echo_show_workload()
        
        # Read hardware counters
        hw_counters = self.read_hardware_performance_counters()
        
        # Calculate hardware metrics
        hw_metrics = self.calculate_hardware_metrics(hw_counters)
        
        # Get system context
        system_info = self.get_system_context()
        
        # Echo Show-specific metrics
        echo_metrics = {
            'video_call_active': 1 if self.video_call_active else 0,
            'alexa_responding': 1 if self.alexa_responding else 0,
            'media_playing': 1 if self.media_playing else 0,
            'smart_home_devices': self.smart_home_devices,
            'weather_alerts': self.weather_alerts,
            'video_call_duration': self.video_call_duration
        }
        
        # Combine all data
        sample = {
            'timestamp': timestamp,
            'workload_state': self.workload_states[self.current_state],
            'workload_intensity': workload_intensity,
            'workload_raw': workload_raw
        }
        sample.update(hw_metrics)
        sample.update(system_info)
        sample.update(echo_metrics)
        
        return sample

    def run_collection(self):
        """Main collection loop"""
        print("=== AMAZON ECHO SHOW HARDWARE PERFORMANCE COUNTERS ===")
        print("Simulating Amazon Echo Show firmware on embedded hardware")
        print(f"Target: {self.samples} samples in {self.duration} seconds")
        print(f"Hardware counters available: {len(self.working_events)}")
        print("-" * 60)
        
        start_time = time.time()
        voice_active_count = 0
        video_active_count = 0
        
        for i in range(self.samples):
            sample_start = time.time()
            
            sample = self.collect_sample()
            self.data.append(sample)
            
            # Count active samples
            if sample['alexa_responding'] > 0:
                voice_active_count += 1
            if sample['video_call_active'] > 0:
                video_active_count += 1
            
            if i % 500 == 0:
                elapsed = time.time() - start_time
                progress = (i / self.samples) * 100
                voice_percent = (voice_active_count / (i + 1)) * 100
                video_percent = (video_active_count / (i + 1)) * 100
                
                print(f"Sample {i:5d}/{self.samples} ({progress:5.1f}%)")
                print(f"  State: {self.current_state:25} | "
                      f"Intensity: {sample['workload_intensity']:5.1f}")
                print(f"  Voice: {voice_percent:5.1f}% | "
                      f"Video Call: {video_percent:5.1f}% | "
                      f"Media: {'Yes' if sample['media_playing'] else 'No'}")
                print(f"  IPC: {sample['hw_instructions_per_cycle']:5.3f} | "
                      f"Temp: {sample['cpu_temp']:4.1f}°C")
                print()
            
            elapsed_sample = time.time() - sample_start
            time.sleep(max(0.001, self.interval - elapsed_sample))
        
        self.save_data()

    def save_data(self):
        """Save data"""
        filename = f"amazon_echo_show_hpc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w') as f:
            headers = self.data[0].keys()
            f.write(','.join(headers) + '\n')
            for sample in self.data:
                row = []
                for h in headers:
                    val = sample[h]
                    if isinstance(val, float):
                        row.append(f"{val:.6f}")
                    else:
                        row.append(str(val))
                f.write(','.join(row) + '\n')
        
        # Final statistics
        total_samples = len(self.data)
        voice_samples = sum(1 for s in self.data if s['alexa_responding'] > 0)
        video_samples = sum(1 for s in self.data if s['video_call_active'] > 0)
        voice_percent = (voice_samples / total_samples) * 100
        video_percent = (video_samples / total_samples) * 100
        avg_intensity = sum(s['workload_intensity'] for s in self.data) / total_samples
        
        print(f"Saved {total_samples} samples to {filename}")
        print(f"Voice interaction samples: {voice_samples}/{total_samples} ({voice_percent:.1f}%)")
        print(f"Video call samples: {video_samples}/{total_samples} ({video_percent:.1f}%)")
        print(f"Average workload intensity: {avg_intensity:.1f}")
        
        # Debug: Show video call distribution
        video_call_states = [s for s in self.data if s['video_call_active'] > 0]
        if video_call_states:
            print(f"Video call samples found: {len(video_call_states)}")
            print(f"Max video call duration: {max(s['video_call_duration'] for s in video_call_states)} samples")

if __name__ == "__main__":
    os.system("echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid > /dev/null 2>&1")
    collector = AmazonEchoShowHPC(samples=20000, duration=600)
    collector.run_collection()
