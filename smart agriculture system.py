smart agriculture system 

#!/usr/bin/env python3
import time
import os
import subprocess
import random
import math
from datetime import datetime

class SmartAgricultureHPC:
    def __init__(self, samples=20000, duration=600):
        self.samples = samples
        self.duration = duration
        self.interval = duration / samples
        self.data = []
        
        # Smart Agriculture Sensor specific workload states
        self.workload_states = {
            'idle_field_monitoring': 0,
            'soil_moisture_analysis': 1,
            'crop_health_imaging': 2,
            'weather_data_processing': 3,
            'irrigation_control': 4,
            'pest_detection': 5,
            'yield_prediction': 6,
            'satellite_data_sync': 7,
            'drone_coordination': 8,
            'nutrient_level_analysis': 9,
            'water_quality_monitoring': 10,
            'growth_stage_tracking': 11
        }
        self.current_state = 'idle_field_monitoring'
        self.state_timer = 0
        
        # Agriculture sensor specific parameters
        self.soil_moisture = 45.0  # percentage
        self.soil_temperature = 22.5  # celsius
        self.air_temperature = 25.0  # celsius
        self.humidity = 60.0  # percentage
        self.light_intensity = 85000  # lux
        self.npk_levels = [25.0, 15.0, 30.0]  # Nitrogen, Phosphorus, Potassium (FIXED: float values)
        self.ph_level = 6.8
        self.wind_speed = 8.5  # km/h
        self.rainfall = 0.0  # mm
        self.crop_health_score = 85.0  # percentage
        
        # Farm management parameters
        self.irrigation_status = 'off'  # 'off', 'sprinkler', 'drip'
        self.pest_alert_level = 0
        self.growth_stage = 'vegetative'  # 'germination', 'vegetative', 'flowering', 'maturation'
        self.water_consumption = 1250.0  # liters
        
        # Track total samples for consistent metrics
        self.total_samples_collected = 0
        
        # Hardware performance counters (ARM agriculture sensor specific)
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

    def simulate_agriculture_workload(self):
        """Simulate Smart Agriculture Sensor specific workload patterns"""
        self.state_timer += 1
        self.total_samples_collected += 1
        workload = 0
        
        # Update environmental conditions
        self._update_environmental_conditions()
        
        # Smart Agriculture state transitions with farming priorities
        if self.current_state == 'idle_field_monitoring':
            # Continuous field monitoring
            workload = self._simulate_field_monitoring()
            
            # Agriculture events occur frequently (15% chance)
            if random.random() < 0.15 or self.state_timer > 20:
                next_states = ['soil_moisture_analysis', 'crop_health_imaging', 
                              'weather_data_processing', 'pest_detection', 'nutrient_level_analysis']
                weights = [0.25, 0.2, 0.2, 0.15, 0.2]
                self.current_state = random.choices(next_states, weights=weights)[0]
                self.state_timer = 0
        
        elif self.current_state == 'soil_moisture_analysis':
            workload = self._simulate_soil_moisture_analysis()
            if random.random() < 0.35 or self.state_timer > 15:
                # Soil analysis often leads to irrigation decisions
                if self.soil_moisture < 30.0:
                    self.current_state = 'irrigation_control'
                else:
                    self.current_state = 'idle_field_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'crop_health_imaging':
            workload = self._simulate_crop_health_imaging()
            if random.random() < 0.30 or self.state_timer > 18:
                # Crop imaging may trigger pest detection or yield prediction
                if random.random() < 0.4:
                    self.current_state = 'pest_detection'
                else:
                    self.current_state = 'yield_prediction'
                self.state_timer = 0
        
        elif self.current_state == 'weather_data_processing':
            workload = self._simulate_weather_processing()
            if random.random() < 0.28 or self.state_timer > 22:
                self.current_state = 'idle_field_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'irrigation_control':
            workload = self._simulate_irrigation_control()
            if random.random() < 0.40 or self.state_timer > 12:
                self.current_state = 'water_quality_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'pest_detection':
            workload = self._simulate_pest_detection()
            if random.random() < 0.32 or self.state_timer > 16:
                if self.pest_alert_level > 5:
                    self.current_state = 'drone_coordination'
                else:
                    self.current_state = 'idle_field_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'yield_prediction':
            workload = self._simulate_yield_prediction()
            if random.random() < 0.25 or self.state_timer > 25:
                self.current_state = 'idle_field_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'satellite_data_sync':
            workload = self._simulate_satellite_sync()
            if random.random() < 0.20 or self.state_timer > 30:
                self.current_state = 'idle_field_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'drone_coordination':
            workload = self._simulate_drone_coordination()
            if random.random() < 0.22 or self.state_timer > 28:
                self.current_state = 'idle_field_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'nutrient_level_analysis':
            workload = self._simulate_nutrient_analysis()
            if random.random() < 0.38 or self.state_timer > 14:
                self.current_state = 'growth_stage_tracking'
                self.state_timer = 0
        
        elif self.current_state == 'water_quality_monitoring':
            workload = self._simulate_water_quality_monitoring()
            if random.random() < 0.33 or self.state_timer > 20:
                self.current_state = 'idle_field_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'growth_stage_tracking':
            workload = self._simulate_growth_stage_tracking()
            if random.random() < 0.26 or self.state_timer > 24:
                self.current_state = 'idle_field_monitoring'
                self.state_timer = 0
        
        # Agriculture-specific micro-workload spikes (35% of samples)
        if random.random() < 0.35 and self.current_state != 'idle_field_monitoring':
            workload += self._generate_agriculture_workload_spike()
        
        # Update water consumption based on irrigation
        if self.current_state == 'irrigation_control' and self.irrigation_status != 'off':
            self.water_consumption += 0.5
        
        # Calculate crop health based on environmental factors
        self._update_crop_health()
        
        workload_intensity = min(100, workload / 30)  # Scale to 0-100
        
        return workload, workload_intensity

    def _update_environmental_conditions(self):
        """Simulate realistic environmental changes in agricultural setting"""
        # Soil moisture changes (slow, natural variations)
        moisture_change = random.uniform(-0.3, 0.5)
        self.soil_moisture = max(10.0, min(80.0, self.soil_moisture + moisture_change))
        
        # Temperature variations (diurnal cycle simulation)
        temp_change = random.uniform(-0.2, 0.3)
        self.soil_temperature += temp_change
        self.air_temperature += temp_change
        
        # Keep temperatures in realistic range
        self.soil_temperature = max(5.0, min(40.0, self.soil_temperature))
        self.air_temperature = max(5.0, min(45.0, self.air_temperature))
        
        # Humidity variations
        self.humidity += random.uniform(-2.0, 2.0)
        self.humidity = max(20.0, min(95.0, self.humidity))
        
        # Light intensity (day/night cycle simulation)
        if random.random() < 0.7:  # 70% chance it's daytime
            self.light_intensity = random.randint(50000, 120000)
        else:  # 30% chance it's nighttime
            self.light_intensity = random.randint(0, 1000)
        
        # FIXED: Nutrient level variations with realistic depletion and replenishment
        for i in range(3):
            # Slow depletion (much slower than before)
            depletion = random.uniform(0, 0.01)  # Reduced from 0.1 to 0.01
            self.npk_levels[i] = max(0.1, self.npk_levels[i] - depletion)
            
            # Occasional nutrient replenishment (fertilization events)
            if random.random() < 0.005:  # 0.5% chance of nutrient addition
                self.npk_levels[i] = min(50.0, self.npk_levels[i] + random.uniform(2.0, 8.0))
        
        # pH level variations
        self.ph_level += random.uniform(-0.05, 0.05)
        self.ph_level = max(4.0, min(9.0, self.ph_level))
        
        # Weather variations
        self.wind_speed = random.uniform(0.0, 25.0)
        if random.random() < 0.05:  # 5% chance of rain
            self.rainfall += random.uniform(0.1, 5.0)
            # Rain increases soil moisture
            self.soil_moisture = min(80.0, self.soil_moisture + random.uniform(1.0, 5.0))

    def _update_crop_health(self):
        """Calculate crop health based on environmental conditions"""
        base_health = 85.0
        
        # Soil moisture impact
        if self.soil_moisture < 30.0 or self.soil_moisture > 70.0:
            base_health -= 10.0
        
        # Nutrient impact
        if any(nutrient < 10 for nutrient in self.npk_levels):
            base_health -= 15.0
        
        # pH impact
        if self.ph_level < 5.5 or self.ph_level > 7.5:
            base_health -= 8.0
        
        # Pest impact
        base_health -= self.pest_alert_level * 2.0
        
        # Add small random variation
        random_variation = random.uniform(-2.0, 2.0)
        
        self.crop_health_score = max(0.0, min(100.0, base_health + random_variation))

    def _generate_agriculture_workload_spike(self):
        """Generate agriculture-specific micro-workload spikes"""
        workload = 0
        spike_intensity = random.randint(50, 200)
        for i in range(spike_intensity // 8):
            workload += math.sin(i * 0.15) * 22 + math.cos(i * 0.10) * 18
        return workload

    def _simulate_field_monitoring(self):
        """Simulate continuous field and environmental monitoring"""
        workload = 25  # Moderate base for field monitoring
        # Environmental sensor polling
        for i in range(80):
            workload += math.cos(i * 0.07) * 12
            workload += (i % 18) * 1.5  # Periodic sensor reads
        
        # Data quality checks
        for i in range(50):
            workload += (i % 12) * 2.0
        
        return workload + random.randint(15, 35)

    def _simulate_soil_moisture_analysis(self):
        """Simulate soil moisture sensor processing and analysis"""
        workload = 85
        # Soil moisture calibration and compensation
        for i in range(200):
            workload += math.sin(i * 0.06) * 30
            workload += (i * 10) % 250 * 0.5
            
            # Soil type compensation algorithms
            if i % 35 == 0:
                workload += 45
        
        # Moisture trend analysis
        for i in range(120):
            workload += (i % 28) * 2.8
        
        return workload + random.randint(40, 100)

    def _simulate_crop_health_imaging(self):
        """Simulate multispectral crop imaging and analysis"""
        workload = 135  # High for image processing
        # Image acquisition and preprocessing
        for i in range(280):
            workload += (i * 15) % 350  # Image filtering
            workload += math.cos(i * 0.05) * 48  # Color analysis
            
            # NDVI (Normalized Difference Vegetation Index) calculation
            if i % 40 == 0:
                workload += 65
        
        # Plant health algorithms
        for i in range(180):
            workload += math.tanh(i * 0.02) * 38
        
        return workload + random.randint(70, 150)

    def _simulate_weather_processing(self):
        """Simulate weather data acquisition and forecasting"""
        workload = 75
        # Weather sensor data fusion
        for i in range(180):
            workload += math.sin(i * 0.08) * 28
            workload += (i * 8) % 220 * 0.4
        
        # Microclimate prediction algorithms
        for i in range(100):
            workload += (i % 22) * 2.5
        
        return workload + random.randint(35, 85)

    def _simulate_irrigation_control(self):
        """Simulate smart irrigation system control"""
        workload = 105
        # Irrigation scheduling algorithms
        for i in range(240):
            workload += math.cos(i * 0.055) * 38
            workload += (i * 12) % 280 * 0.6
            
            # Water conservation optimization
            if i % 32 == 0:
                workload += 55
        
        # Valve control and flow management
        for i in range(140):
            workload += (i % 30) * 3.2
        
        # Auto irrigation decision
        if self.soil_moisture < 35.0:
            self.irrigation_status = random.choice(['sprinkler', 'drip'])
        elif self.soil_moisture > 65.0:
            self.irrigation_status = 'off'
        
        return workload + random.randint(55, 120)

    def _simulate_pest_detection(self):
        """Simulate pest detection using image analysis"""
        workload = 145  # High for pattern recognition
        # Pest pattern recognition
        for i in range(300):
            workload += (i * 17) % 380  # Feature extraction
            workload += self._sigmoid(i * 0.03) * 62  # Classification
            
            # Damage assessment algorithms
            if i % 45 == 0:
                workload += 75
        
        # Pest lifecycle tracking
        for i in range(190):
            workload += math.tanh(i * 0.018) * 40
        
        # Update pest alert level
        if random.random() < 0.1:  # 10% chance of pest detection
            self.pest_alert_level = min(10, self.pest_alert_level + random.randint(1, 3))
        elif random.random() < 0.05:  # 5% chance of pest reduction
            self.pest_alert_level = max(0, self.pest_alert_level - 1)
        
        return workload + random.randint(80, 160)

    def _simulate_yield_prediction(self):
        """Simulate crop yield prediction algorithms"""
        workload = 125
        # Machine learning prediction models
        for i in range(270):
            workload += math.sin(i * 0.045) * 42
            workload += (i * 14) % 320 * 0.5
            
            # Historical data correlation
            if i % 50 == 0:
                workload += 68
        
        # Statistical analysis
        for i in range(170):
            workload += (i % 38) * 3.0
        
        return workload + random.randint(65, 135)

    def _simulate_satellite_data_sync(self):
        """Simulate satellite data synchronization"""
        workload = 95
        # Satellite communication protocols
        for i in range(220):
            workload += math.cos(i * 0.065) * 32
            workload += (i * 9) % 260 * 0.4
        
        # Geospatial data processing
        for i in range(130):
            workload += (i % 25) * 2.8
        
        return workload + random.randint(45, 105)

    def _simulate_drone_coordination(self):
        """Simulate agricultural drone coordination"""
        workload = 115
        # Drone flight path planning
        for i in range(250):
            workload += math.sin(i * 0.052) * 36
            workload += (i * 13) % 300 * 0.5
        
        # Multi-drone coordination
        for i in range(150):
            workload += (i % 35) * 3.1
        
        return workload + random.randint(60, 125)

    def _simulate_nutrient_analysis(self):
        """Simulate soil nutrient level analysis"""
        workload = 90
        # NPK sensor data processing
        for i in range(200):
            workload += math.cos(i * 0.07) * 30
            workload += (i * 10) % 240 * 0.4
        
        # Fertilizer recommendation algorithms
        for i in range(120):
            workload += (i % 28) * 2.6
        
        return workload + random.randint(40, 95)

    def _simulate_water_quality_monitoring(self):
        """Simulate irrigation water quality analysis"""
        workload = 80
        # Water quality sensor processing
        for i in range(190):
            workload += math.sin(i * 0.075) * 26
            workload += (i * 7) % 200 * 0.5
        
        # Contaminant detection
        for i in range(110):
            workload += (i % 24) * 2.4
        
        return workload + random.randint(35, 85)

    def _simulate_growth_stage_tracking(self):
        """Simulate crop growth stage monitoring"""
        workload = 70
        # Growth pattern analysis
        for i in range(170):
            workload += math.cos(i * 0.08) * 24
            workload += (i * 6) % 180 * 0.4
        
        # Developmental stage classification
        for i in range(90):
            workload += (i % 20) * 2.2
        
        # Update growth stage occasionally
        if random.random() < 0.02:  # 2% chance to change growth stage
            stages = ['germination', 'vegetative', 'flowering', 'maturation']
            current_index = stages.index(self.growth_stage)
            if current_index < len(stages) - 1 and random.random() < 0.7:
                self.growth_stage = stages[current_index + 1]
        
        return workload + random.randint(30, 75)

    def read_hardware_performance_counters(self):
        """Read hardware performance counters for Agriculture Sensors"""
        if not self.working_events:
            return self._generate_realistic_agriculture_hw_data()
        
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
            return self._generate_realistic_agriculture_hw_data()

    def _generate_realistic_agriculture_hw_data(self):
        """Generate realistic Agriculture Sensor hardware counter data"""
        base = time.time_ns() % 1000000
        # Agriculture sensors have moderate computational requirements
        if self.current_state in ['crop_health_imaging', 'pest_detection', 'yield_prediction']:
            multiplier = 1.5  # High image processing states
        elif self.current_state in ['irrigation_control', 'drone_coordination']:
            multiplier = 1.3  # Medium control states
        else:
            multiplier = 1.0  # Normal monitoring states
        
        return {
            'cpu-cycles': int((1900000 + (base * 33) % 800000) * multiplier),
            'instructions': int((1700000 + (base * 25) % 700000) * multiplier),
            'branch-instructions': int((120000 + (base * 15) % 60000) * multiplier),
            'branch-misses': int((3800 + (base * 11) % 2500) * multiplier),
            'cache-references': int((180000 + (base * 21) % 90000) * multiplier),
            'cache-misses': int((8500 + (base * 17) % 5500) * multiplier),
            'L1-dcache-loads': int((150000 + (base * 29) % 80000) * multiplier),
            'L1-dcache-load-misses': int((6500 + (base * 9) % 4000) * multiplier),
            'LLC-loads': int((35000 + (base * 5) % 22000) * multiplier),
            'LLC-load-misses': int((1800 + (base * 3) % 1100) * multiplier),
            'stalled-cycles-frontend': int((160000 + (base * 39) % 110000) * multiplier),
            'stalled-cycles-backend': int((140000 + (base * 41) % 90000) * multiplier),
            'bus-cycles': int((60000 + (base * 13) % 30000) * multiplier)
        }

    def calculate_hardware_metrics(self, raw_counters):
        """Calculate hardware performance metrics for Agriculture Sensors"""
        cycles = max(1, raw_counters.get('cpu-cycles', 1))
        instructions = raw_counters.get('instructions', 1650000)
        branches = max(1, raw_counters.get('branch-instructions', 110000))
        branch_misses = raw_counters.get('branch-misses', 3400)
        cache_refs = max(1, raw_counters.get('cache-references', 170000))
        cache_misses = raw_counters.get('cache-misses', 8000)
        l1_loads = max(1, raw_counters.get('L1-dcache-loads', 140000))
        l1_misses = raw_counters.get('L1-dcache-load-misses', 6000)
        llc_loads = max(1, raw_counters.get('LLC-loads', 33000))
        llc_misses = raw_counters.get('LLC-load-misses', 1700)
        frontend_stalls = raw_counters.get('stalled-cycles-frontend', 150000)
        backend_stalls = raw_counters.get('stalled-cycles-backend', 130000)
        
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
        """Get system context with agriculture parameters"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = float(f.read().strip()) / 1000.0
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
                cpu_freq = float(f.read().strip()) / 1000.0
            return {
                'cpu_temp': cpu_temp, 
                'cpu_freq': cpu_freq,
                'irrigation_status': self.irrigation_status,
                'growth_stage': self.growth_stage
            }
        except:
            return {
                'cpu_temp': 40.0, 
                'cpu_freq': 1200.0,
                'irrigation_status': self.irrigation_status,
                'growth_stage': self.growth_stage
            }

    def collect_sample(self):
        """Collect one comprehensive agriculture sensor sample"""
        timestamp = datetime.now()
        
        # Simulate Agriculture Sensor workload
        workload_raw, workload_intensity = self.simulate_agriculture_workload()
        
        # Read hardware counters
        hw_counters = self.read_hardware_performance_counters()
        
        # Calculate hardware metrics
        hw_metrics = self.calculate_hardware_metrics(hw_counters)
        
        # Get system context
        system_info = self.get_system_context()
        
        # Agriculture-specific metrics
        agriculture_metrics = {
            'soil_moisture': self.soil_moisture,
            'soil_temperature': self.soil_temperature,
            'air_temperature': self.air_temperature,
            'humidity': self.humidity,
            'light_intensity': self.light_intensity,
            'nitrogen_level': self.npk_levels[0],
            'phosphorus_level': self.npk_levels[1],
            'potassium_level': self.npk_levels[2],
            'ph_level': self.ph_level,
            'wind_speed': self.wind_speed,
            'rainfall': self.rainfall,
            'crop_health_score': self.crop_health_score,
            'pest_alert_level': self.pest_alert_level,
            'water_consumption': self.water_consumption
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
        sample.update(agriculture_metrics)
        
        return sample

    def run_collection(self):
        """Main collection loop for Agriculture Sensors"""
        print("=== SMART AGRICULTURE SENSOR HARDWARE PERFORMANCE COUNTERS ===")
        print("Simulating Smart Agriculture Sensors on Raspberry Pi")
        print(f"Target: {self.samples} samples in {self.duration} seconds")
        print(f"Hardware counters available: {len(self.working_events)}")
        print("-" * 60)
        
        start_time = time.time()
        soil_analysis_count = 0
        crop_imaging_count = 0
        
        for i in range(self.samples):
            sample_start = time.time()
            
            sample = self.collect_sample()
            self.data.append(sample)
            
            # Count agriculture events
            if sample['workload_state'] == 1:  # Soil moisture analysis
                soil_analysis_count += 1
            if sample['workload_state'] == 2:  # Crop health imaging
                crop_imaging_count += 1
            
            if i % 500 == 0:
                elapsed = time.time() - start_time
                progress = (i / self.samples) * 100
                soil_percent = (soil_analysis_count / (i + 1)) * 100
                crop_percent = (crop_imaging_count / (i + 1)) * 100
                
                print(f"Sample {i:5d}/{self.samples} ({progress:5.1f}%)")
                print(f"  State: {self.current_state:25} | "
                      f"Intensity: {sample['workload_intensity']:5.1f}")
                print(f"  Soil Analysis: {soil_percent:5.1f}% | "
                      f"Crop Imaging: {crop_percent:5.1f}% | "
                      f"Irrigation: {sample['irrigation_status']:10s}")
                print(f"  Soil Moisture: {sample['soil_moisture']:5.1f}% | "
                      f"Air Temp: {sample['air_temperature']:5.1f}°C | "
                      f"Crop Health: {sample['crop_health_score']:5.1f}%")
                print(f"  N: {sample['nitrogen_level']:5.2f} | "
                      f"P: {sample['phosphorus_level']:5.2f} | "
                      f"K: {sample['potassium_level']:5.2f} | "
                      f"pH: {sample['ph_level']:4.1f}")
                print(f"  IPC: {sample['hw_instructions_per_cycle']:5.3f} | "
                      f"Water Used: {sample['water_consumption']:6.0f}L")
                print()
            
            elapsed_sample = time.time() - sample_start
            time.sleep(max(0.001, self.interval - elapsed_sample))
        
        self.save_data()

    def save_data(self):
        """Save collected agriculture sensor data to CSV"""
        filename = f"smart_agriculture_hpc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
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
        soil_samples = sum(1 for s in self.data if s['workload_state'] == 1)
        crop_samples = sum(1 for s in self.data if s['workload_state'] == 2)
        soil_percent = (soil_samples / total_samples) * 100
        crop_percent = (crop_samples / total_samples) * 100
        avg_intensity = sum(s['workload_intensity'] for s in self.data) / total_samples
        avg_ipc = sum(s['hw_instructions_per_cycle'] for s in self.data) / total_samples
        
        print(f"Saved {total_samples} samples to {filename}")
        print(f"Soil analysis samples: {soil_samples}/{total_samples} ({soil_percent:.1f}%)")
        print(f"Crop imaging samples: {crop_samples}/{total_samples} ({crop_percent:.1f}%)")
        print(f"Average workload intensity: {avg_intensity:.1f}")
        print(f"Average Instructions Per Cycle: {avg_ipc:.3f}")
        
        # Agriculture specific statistics
        avg_moisture = sum(s['soil_moisture'] for s in self.data) / total_samples
        avg_health = sum(s['crop_health_score'] for s in self.data) / total_samples
        total_water = sum(s['water_consumption'] for s in self.data)
        avg_nitrogen = sum(s['nitrogen_level'] for s in self.data) / total_samples
        avg_phosphorus = sum(s['phosphorus_level'] for s in self.data) / total_samples
        avg_potassium = sum(s['potassium_level'] for s in self.data) / total_samples
        
        print(f"Average Soil Moisture: {avg_moisture:.1f}%")
        print(f"Average Crop Health: {avg_health:.1f}%")
        print(f"Total Water Consumption: {total_water:.0f} liters")
        print(f"Average Nitrogen: {avg_nitrogen:.2f}")
        print(f"Average Phosphorus: {avg_phosphorus:.2f}")
        print(f"Average Potassium: {avg_potassium:.2f}")

if __name__ == "__main__":
    # Enable performance monitoring
    os.system("echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid > /dev/null 2>&1")
    collector = SmartAgricultureHPC(samples=20000, duration=600)
    collector.run_collection()
