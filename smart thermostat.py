smart thermostat 

#!/usr/bin/env python3
import time
import os
import subprocess
import random
import math
from datetime import datetime

class SmartThermostatHPC:
    def __init__(self, samples=20000, duration=600):
        self.samples = samples
        self.duration = duration
        self.interval = duration / samples
        self.data = []
        
        # Smart Thermostat specific workload states
        self.workload_states = {
            'idle_temperature_monitoring': 0,
            'climate_algorithm_calculation': 1,
            'occupancy_detection': 2,
            'weather_data_processing': 3,
            'schedule_optimization': 4,
            'hvac_control_signaling': 5,
            'energy_usage_analytics': 6,
            'remote_access_processing': 7,
            'firmware_ota_check': 8,
            'sensor_data_fusion': 9,
            'learning_algorithm_update': 10,
            'humidity_control': 11
        }
        self.current_state = 'idle_temperature_monitoring'
        self.state_timer = 0
        
        # Thermostat-specific parameters
        self.current_temp = 21.5
        self.target_temp = 22.0
        self.humidity = 45
        self.occupancy = True
        self.hvac_mode = 'heat'  # 'heat', 'cool', 'off'
        self.fan_speed = 'auto'
        self.outdoor_temp = 15.0
        self.energy_usage = 2.3  # kWh
        
        # Learning algorithm parameters
        self.learning_enabled = True
        self.comfort_profile_active = True
        self.away_mode = False
        
        # Hardware performance counters (ARM thermostat specific)
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

    def simulate_thermostat_workload(self):
        """Simulate Smart Thermostat specific workload patterns"""
        self.state_timer += 1
        workload = 0
        
        # Update environmental conditions
        self._update_environmental_conditions()
        
        # Smart Thermostat state transitions with HVAC priorities
        if self.current_state == 'idle_temperature_monitoring':
            # Continuous temperature and humidity monitoring
            workload = self._simulate_temperature_monitoring()
            
            # Thermostat activations occur frequently (15% chance)
            if random.random() < 0.15 or self.state_timer > 20:
                next_states = ['climate_algorithm_calculation', 'occupancy_detection', 
                              'weather_data_processing', 'sensor_data_fusion', 'hvac_control_signaling']
                weights = [0.3, 0.25, 0.2, 0.15, 0.1]
                self.current_state = random.choices(next_states, weights=weights)[0]
                self.state_timer = 0
        
        elif self.current_state == 'climate_algorithm_calculation':
            workload = self._simulate_climate_algorithm()
            if random.random() < 0.40 or self.state_timer > 15:
                # Climate calculations often lead to HVAC control
                if abs(self.current_temp - self.target_temp) > 0.5:
                    self.current_state = 'hvac_control_signaling'
                else:
                    self.current_state = 'idle_temperature_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'occupancy_detection':
            workload = self._simulate_occupancy_detection()
            if random.random() < 0.35 or self.state_timer > 12:
                # Occupancy changes may trigger schedule updates
                if not self.occupancy and random.random() < 0.6:
                    self.current_state = 'schedule_optimization'
                else:
                    self.current_state = 'idle_temperature_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'weather_data_processing':
            workload = self._simulate_weather_processing()
            if random.random() < 0.30 or self.state_timer > 18:
                self.current_state = 'climate_algorithm_calculation'
                self.state_timer = 0
        
        elif self.current_state == 'schedule_optimization':
            workload = self._simulate_schedule_optimization()
            if random.random() < 0.25 or self.state_timer > 25:
                self.current_state = 'idle_temperature_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'hvac_control_signaling':
            workload = self._simulate_hvac_control()
            if random.random() < 0.45 or self.state_timer > 8:
                self.current_state = 'energy_usage_analytics'
                self.state_timer = 0
        
        elif self.current_state == 'energy_usage_analytics':
            workload = self._simulate_energy_analytics()
            if random.random() < 0.28 or self.state_timer > 15:
                self.current_state = 'idle_temperature_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'remote_access_processing':
            workload = self._simulate_remote_access()
            if random.random() < 0.22 or self.state_timer > 10:
                self.current_state = 'idle_temperature_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'firmware_ota_check':
            workload = self._simulate_firmware_update()
            if random.random() < 0.12 or self.state_timer > 45:
                self.current_state = 'idle_temperature_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'sensor_data_fusion':
            workload = self._simulate_sensor_fusion()
            if random.random() < 0.32 or self.state_timer > 20:
                self.current_state = 'climate_algorithm_calculation'
                self.state_timer = 0
        
        elif self.current_state == 'learning_algorithm_update':
            workload = self._simulate_learning_update()
            if random.random() < 0.18 or self.state_timer > 60:
                self.current_state = 'idle_temperature_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'humidity_control':
            workload = self._simulate_humidity_control()
            if random.random() < 0.25 or self.state_timer > 15:
                self.current_state = 'idle_temperature_monitoring'
                self.state_timer = 0
        
        # HVAC-specific micro-workload spikes (35% of samples)
        if random.random() < 0.35 and self.current_state != 'idle_temperature_monitoring':
            workload += self._generate_hvac_workload_spike()
        
        # Update energy usage based on HVAC activity
        if self.current_state == 'hvac_control_signaling':
            self.energy_usage += 0.01
        
        workload_intensity = min(100, workload / 30)  # Scale to 0-100
        
        return workload, workload_intensity

    def _update_environmental_conditions(self):
        """Simulate realistic environmental changes"""
        # Temperature fluctuations
        temp_change = random.uniform(-0.2, 0.2)
        self.current_temp += temp_change
        
        # Outdoor temperature variations (slower changes)
        if random.random() < 0.1:
            self.outdoor_temp += random.uniform(-0.5, 0.5)
        
        # Humidity variations
        self.humidity += random.randint(-2, 2)
        self.humidity = max(30, min(70, self.humidity))
        
        # Occupancy changes (less frequent)
        if random.random() < 0.02:
            self.occupancy = not self.occupancy
        
        # Target temperature adjustments (user changes)
        if random.random() < 0.01:
            self.target_temp += random.randint(-1, 1)
            self.target_temp = max(18, min(26, self.target_temp))

    def _generate_hvac_workload_spike(self):
        """Generate HVAC-specific micro-workload spikes"""
        workload = 0
        spike_intensity = random.randint(40, 180)
        for i in range(spike_intensity // 8):
            workload += math.sin(i * 0.12) * 18 + math.cos(i * 0.09) * 12
        return workload

    def _simulate_temperature_monitoring(self):
        """Simulate continuous temperature and sensor monitoring"""
        workload = 15  # Low base for monitoring
        # Temperature sensor polling
        for i in range(60):
            workload += math.cos(i * 0.08) * 8
            workload += (i % 15) * 1.2  # Periodic sensor reads
        
        # Environmental quality monitoring
        for i in range(40):
            workload += (i % 10) * 1.5
        
        return workload + random.randint(10, 25)

    def _simulate_climate_algorithm(self):
        """Simulate advanced climate control algorithms"""
        workload = 85
        # PID control calculations
        for i in range(200):
            # Proportional term
            workload += abs(self.current_temp - self.target_temp) * 2.5
            # Integral term simulation
            workload += math.sin(i * 0.06) * 25
            # Derivative term simulation
            workload += (i * 7) % 180 * 0.4
            
            # Adaptive algorithm adjustments
            if i % 40 == 0:
                workload += 35
        
        # Energy efficiency calculations
        for i in range(120):
            workload += (i % 25) * 2.8
        
        return workload + random.randint(40, 100)

    def _simulate_occupancy_detection(self):
        """Simulate occupancy sensing and pattern recognition"""
        workload = 75
        # Motion detection and pattern analysis
        for i in range(180):
            workload += math.sin(i * 0.07) * 30
            workload += (i * 9) % 220 * 0.5
            
            # Presence detection algorithms
            if i % 35 == 0:
                workload += 45
        
        # Behavioral pattern learning
        for i in range(100):
            workload += (i % 20) * 3.2
        
        return workload + random.randint(35, 85)

    def _simulate_weather_processing(self):
        """Simulate weather data acquisition and processing"""
        workload = 65
        # Weather API communication and parsing
        for i in range(160):
            workload += math.cos(i * 0.05) * 22
            workload += (i * 6) % 200 * 0.4
        
        # Forecast integration
        for i in range(90):
            workload += (i % 18) * 2.5
        
        return workload + random.randint(30, 70)

    def _simulate_schedule_optimization(self):
        """Simulate schedule learning and optimization"""
        workload = 95
        # Schedule optimization algorithms
        for i in range(220):
            workload += math.sin(i * 0.04) * 35
            workload += (i * 8) % 260 * 0.5
            
            # Pattern recognition for schedule learning
            if i % 50 == 0:
                workload += 55
        
        # Energy cost optimization
        for i in range(140):
            workload += (i % 28) * 3.0
        
        return workload + random.randint(45, 110)

    def _simulate_hvac_control(self):
        """Simulate HVAC system control and signaling"""
        workload = 105
        # HVAC equipment control logic
        for i in range(240):
            workload += (i * 12) % 300  # Control signal generation
            workload += math.cos(i * 0.06) * 40  # System state management
            
            # Compressor and fan control
            if i % 45 == 0:
                workload += 60
        
        # Safety and limit checking
        for i in range(120):
            workload += (i % 25) * 3.5
        
        return workload + random.randint(50, 120)

    def _simulate_energy_analytics(self):
        """Simulate energy usage analysis and reporting"""
        workload = 70
        # Energy consumption calculations
        for i in range(180):
            workload += math.sin(i * 0.08) * 25
            workload += (i * 5) % 200 * 0.6
        
        # Cost analysis and reporting
        for i in range(100):
            workload += (i % 22) * 2.8
        
        return workload + random.randint(35, 80)

    def _simulate_remote_access(self):
        """Simulate cloud connectivity and remote access"""
        workload = 80
        # Cloud communication protocols
        for i in range(200):
            workload += math.cos(i * 0.07) * 28
            workload += (i * 7) % 240 * 0.4
        
        # Data synchronization
        for i in range(110):
            workload += (i % 20) * 3.0
        
        return workload + random.randint(40, 90)

    def _simulate_firmware_update(self):
        """Simulate OTA firmware update processing"""
        workload = 115
        # Firmware verification and installation
        for i in range(250):
            workload += (i * 14) % 320  # Security verification
            workload += math.sin(i * 0.05) * 45  # Update process
        
        # System integrity checks
        for i in range(150):
            workload += (i % 30) * 3.2
        
        return workload + random.randint(60, 130)

    def _simulate_sensor_fusion(self):
        """Simulate multi-sensor data fusion"""
        workload = 90
        # Multiple sensor data integration
        sensors = ['temperature', 'humidity', 'motion', 'light', 'air_quality']
        for sensor_idx, sensor in enumerate(sensors):
            for i in range(100):
                workload += math.sin(sensor_idx * 0.8 + i * 0.06) * 20
                workload += (sensor_idx * 18 + i * 4) % 180 * 0.3
        
        # Data correlation analysis
        for i in range(120):
            workload += (i % 25) * 2.5
        
        return workload + random.randint(45, 100)

    def _simulate_learning_update(self):
        """Simulate machine learning model updates"""
        workload = 125
        # Neural network training for behavior learning
        for i in range(280):
            workload += (i * 16) % 350  # Gradient calculations
            workload += self._sigmoid(i * 0.03) * 50  # Activation functions
            
            # Model weight updates
            if i % 60 == 0:
                workload += 65
        
        # Pattern recognition algorithms
        for i in range(180):
            workload += math.tanh(i * 0.02) * 35
        
        return workload + random.randint(70, 140)

    def _simulate_humidity_control(self):
        """Simulate humidity monitoring and control"""
        workload = 60
        # Humidity sensor processing
        for i in range(150):
            workload += math.cos(i * 0.09) * 20
            workload += (i % 30) * 1.8
        
        # Dehumidifier/humidifier control logic
        for i in range(80):
            workload += (i % 16) * 2.2
        
        return workload + random.randint(30, 65)

    def read_hardware_performance_counters(self):
        """Read hardware performance counters for Smart Thermostat"""
        if not self.working_events:
            return self._generate_realistic_thermostat_hw_data()
        
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
            return self._generate_realistic_thermostat_hw_data()

    def _generate_realistic_thermostat_hw_data(self):
        """Generate realistic Smart Thermostat hardware counter data"""
        base = time.time_ns() % 1000000
        # Thermostats have optimized power profiles
        if self.current_state in ['climate_algorithm_calculation', 'learning_algorithm_update']:
            multiplier = 1.4  # High computational states
        elif self.current_state in ['hvac_control_signaling', 'schedule_optimization']:
            multiplier = 1.2  # Medium control states
        else:
            multiplier = 0.8  # Low power monitoring states
        
        return {
            'cpu-cycles': int((1750000 + (base * 29) % 700000) * multiplier),
            'instructions': int((1550000 + (base * 21) % 600000) * multiplier),
            'branch-instructions': int((95000 + (base * 9) % 50000) * multiplier),
            'branch-misses': int((3000 + (base * 5) % 1500) * multiplier),
            'cache-references': int((155000 + (base * 15) % 80000) * multiplier),
            'cache-misses': int((7000 + (base * 11) % 4000) * multiplier),
            'L1-dcache-loads': int((135000 + (base * 23) % 70000) * multiplier),
            'L1-dcache-load-misses': int((5000 + (base * 3) % 2500) * multiplier),
            'LLC-loads': int((28000 + (base * 2) % 18000) * multiplier),
            'LLC-load-misses': int((1400 + (base * 1) % 800) * multiplier),
            'stalled-cycles-frontend': int((135000 + (base * 33) % 90000) * multiplier),
            'stalled-cycles-backend': int((115000 + (base * 35) % 70000) * multiplier),
            'bus-cycles': int((48000 + (base * 9) % 24000) * multiplier)
        }

    def calculate_hardware_metrics(self, raw_counters):
        """Calculate hardware performance metrics for Smart Thermostat"""
        cycles = max(1, raw_counters.get('cpu-cycles', 1))
        instructions = raw_counters.get('instructions', 1450000)
        branches = max(1, raw_counters.get('branch-instructions', 80000))
        branch_misses = raw_counters.get('branch-misses', 2400)
        cache_refs = max(1, raw_counters.get('cache-references', 135000))
        cache_misses = raw_counters.get('cache-misses', 6000)
        l1_loads = max(1, raw_counters.get('L1-dcache-loads', 115000))
        l1_misses = raw_counters.get('L1-dcache-load-misses', 4500)
        llc_loads = max(1, raw_counters.get('LLC-loads', 26000))
        llc_misses = raw_counters.get('LLC-load-misses', 1200)
        frontend_stalls = raw_counters.get('stalled-cycles-frontend', 105000)
        backend_stalls = raw_counters.get('stalled-cycles-backend', 85000)
        
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
        """Get system context with thermostat parameters"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = float(f.read().strip()) / 1000.0
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
                cpu_freq = float(f.read().strip()) / 1000.0
            return {
                'cpu_temp': cpu_temp, 
                'cpu_freq': cpu_freq,
                'hvac_mode': self.hvac_mode,
                'learning_enabled': 1 if self.learning_enabled else 0
            }
        except:
            return {
                'cpu_temp': 38.0, 
                'cpu_freq': 1000.0,
                'hvac_mode': self.hvac_mode,
                'learning_enabled': 1 if self.learning_enabled else 0
            }

    def collect_sample(self):
        """Collect one comprehensive thermostat sample"""
        timestamp = datetime.now()
        
        # Simulate Smart Thermostat workload
        workload_raw, workload_intensity = self.simulate_thermostat_workload()
        
        # Read hardware counters
        hw_counters = self.read_hardware_performance_counters()
        
        # Calculate hardware metrics
        hw_metrics = self.calculate_hardware_metrics(hw_counters)
        
        # Get system context
        system_info = self.get_system_context()
        
        # Thermostat-specific metrics
        thermostat_metrics = {
            'current_temperature': self.current_temp,
            'target_temperature': self.target_temp,
            'humidity_level': self.humidity,
            'occupancy_detected': 1 if self.occupancy else 0,
            'outdoor_temperature': self.outdoor_temp,
            'energy_usage_kwh': self.energy_usage,
            'away_mode': 1 if self.away_mode else 0,
            'comfort_profile': 1 if self.comfort_profile_active else 0
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
        sample.update(thermostat_metrics)
        
        return sample

    def run_collection(self):
        """Main collection loop for Smart Thermostat"""
        print("=== SMART THERMOSTAT HARDWARE PERFORMANCE COUNTERS ===")
        print("Simulating Smart Thermostat on Raspberry Pi")
        print(f"Target: {self.samples} samples in {self.duration} seconds")
        print(f"Hardware counters available: {len(self.working_events)}")
        print("-" * 60)
        
        start_time = time.time()
        climate_algo_count = 0
        hvac_control_count = 0
        
        for i in range(self.samples):
            sample_start = time.time()
            
            sample = self.collect_sample()
            self.data.append(sample)
            
            # Count thermostat events
            if sample['workload_state'] == 1:  # Climate algorithm
                climate_algo_count += 1
            if sample['workload_state'] == 5:  # HVAC control
                hvac_control_count += 1
            
            if i % 500 == 0:
                elapsed = time.time() - start_time
                progress = (i / self.samples) * 100
                climate_percent = (climate_algo_count / (i + 1)) * 100
                hvac_percent = (hvac_control_count / (i + 1)) * 100
                
                print(f"Sample {i:5d}/{self.samples} ({progress:5.1f}%)")
                print(f"  State: {self.current_state:28} | "
                      f"Intensity: {sample['workload_intensity']:5.1f}")
                print(f"  Climate Algo: {climate_percent:5.1f}% | "
                      f"HVAC Control: {hvac_percent:5.1f}% | "
                      f"Mode: {sample['hvac_mode']:4s}")
                print(f"  Temp: {sample['current_temperature']:4.1f}°C | "
                      f"Target: {sample['target_temperature']:4.1f}°C | "
                      f"Humidity: {sample['humidity_level']:2d}%")
                print(f"  IPC: {sample['hw_instructions_per_cycle']:5.3f} | "
                      f"Energy: {sample['energy_usage_kwh']:5.2f} kWh")
                print()
            
            elapsed_sample = time.time() - sample_start
            time.sleep(max(0.001, self.interval - elapsed_sample))
        
        self.save_data()

    def save_data(self):
        """Save collected thermostat data to CSV"""
        filename = f"smart_thermostat_hpc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
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
        climate_samples = sum(1 for s in self.data if s['workload_state'] == 1)
        hvac_samples = sum(1 for s in self.data if s['workload_state'] == 5)
        climate_percent = (climate_samples / total_samples) * 100
        hvac_percent = (hvac_samples / total_samples) * 100
        avg_intensity = sum(s['workload_intensity'] for s in self.data) / total_samples
        avg_ipc = sum(s['hw_instructions_per_cycle'] for s in self.data) / total_samples
        
        print(f"Saved {total_samples} samples to {filename}")
        print(f"Climate algorithm samples: {climate_samples}/{total_samples} ({climate_percent:.1f}%)")
        print(f"HVAC control samples: {hvac_samples}/{total_samples} ({hvac_percent:.1f}%)")
        print(f"Average workload intensity: {avg_intensity:.1f}")
        print(f"Average Instructions Per Cycle: {avg_ipc:.3f}")
        
        # Thermostat specific statistics
        avg_temp = sum(s['current_temperature'] for s in self.data) / total_samples
        avg_energy = sum(s['energy_usage_kwh'] for s in self.data) / total_samples
        print(f"Average Temperature: {avg_temp:.1f}°C")
        print(f"Total Energy Usage: {avg_energy * total_samples / 3600:.2f} kWh")

if __name__ == "__main__":
    # Enable performance monitoring
    os.system("echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid > /dev/null 2>&1")
    collector = SmartThermostatHPC(samples=20000, duration=600)
    collector.run_collection()
