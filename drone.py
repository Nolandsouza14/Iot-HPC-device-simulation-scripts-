drone 

#!/usr/bin/env python3
import time
import os
import subprocess
import random
import math
from datetime import datetime

class DroneFlightControllerHPC:
    def __init__(self, samples=20000, duration=600):
        self.samples = samples
        self.duration = duration
        self.interval = duration / samples
        self.data = []
        
        # Drone Flight Controller specific workload states
        self.workload_states = {
            'idle_ground_station': 0,
            'sensor_fusion_processing': 1,
            'flight_stabilization': 2,
            'gps_navigation': 3,
            'obstacle_avoidance': 4,
            'battery_management': 5,
            'telemetry_transmission': 6,
            'camera_gimbal_control': 7,
            'autonomous_flight': 8,
            'emergency_landing': 9,
            'motor_control': 10,
            'waypoint_navigation': 11,
            'vision_processing': 12
        }
        self.current_state = 'idle_ground_station'
        self.state_timer = 0
        
        # Drone flight controller specific parameters
        self.altitude = 50.0  # meters
        self.battery_level = 85.0  # percentage - START WITH REALISTIC VALUE
        self.gps_satellites = 12
        self.flight_mode = 'stabilize'  # 'stabilize', 'loiter', 'auto', 'rtl'
        self.motor_speeds = [1200, 1200, 1200, 1200]  # RPM for 4 motors
        self.attitude = [0.1, 0.05, 45.0]  # roll, pitch, yaw in degrees
        self.gps_coords = [37.7749, -122.4194]  # lat, long
        self.obstacle_distance = 25.0  # meters
        self.signal_strength = -65  # dBm
        
        # Flight control parameters
        self.target_altitude = 50.0
        self.target_heading = 45.0
        self.airspeed = 8.5  # m/s
        self.vibration_level = 0.02
        
        # Track total samples for battery calculation
        self.total_samples_collected = 0
        
        # Hardware performance counters (ARM flight controller specific)
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

    def simulate_drone_workload(self):
        """Simulate Drone Flight Controller specific workload patterns"""
        self.state_timer += 1
        self.total_samples_collected += 1
        workload = 0
        
        # Update flight dynamics
        self._update_flight_dynamics()
        
        # Drone Flight Controller state transitions with flight priorities
        if self.current_state == 'idle_ground_station':
            # Ground station monitoring
            workload = self._simulate_ground_station()
            
            # Flight activations occur frequently when armed (20% chance)
            if random.random() < 0.20 or self.state_timer > 15:
                next_states = ['sensor_fusion_processing', 'flight_stabilization', 
                              'gps_navigation', 'battery_management', 'telemetry_transmission']
                weights = [0.25, 0.3, 0.2, 0.1, 0.15]
                self.current_state = random.choices(next_states, weights=weights)[0]
                self.state_timer = 0
        
        elif self.current_state == 'sensor_fusion_processing':
            workload = self._simulate_sensor_fusion()
            if random.random() < 0.45 or self.state_timer > 8:
                # Sensor fusion leads to stabilization or navigation
                if random.random() < 0.6:
                    self.current_state = 'flight_stabilization'
                else:
                    self.current_state = 'gps_navigation'
                self.state_timer = 0
        
        elif self.current_state == 'flight_stabilization':
            workload = self._simulate_flight_stabilization()
            if random.random() < 0.35 or self.state_timer > 12:
                # Stabilization can lead to autonomous flight or continue
                if random.random() < 0.3:
                    self.current_state = 'autonomous_flight'
                else:
                    self.current_state = 'sensor_fusion_processing'
                self.state_timer = 0
        
        elif self.current_state == 'gps_navigation':
            workload = self._simulate_gps_navigation()
            if random.random() < 0.30 or self.state_timer > 18:
                if random.random() < 0.4:
                    self.current_state = 'waypoint_navigation'
                else:
                    self.current_state = 'sensor_fusion_processing'
                self.state_timer = 0
        
        elif self.current_state == 'obstacle_avoidance':
            workload = self._simulate_obstacle_avoidance()
            if random.random() < 0.40 or self.state_timer > 6:
                # Obstacle avoidance is critical and fast
                if self.obstacle_distance < 5.0:
                    self.current_state = 'emergency_landing'
                else:
                    self.current_state = 'flight_stabilization'
                self.state_timer = 0
        
        elif self.current_state == 'battery_management':
            workload = self._simulate_battery_management()
            if random.random() < 0.25 or self.state_timer > 20:
                if self.battery_level < 20.0:
                    self.current_state = 'emergency_landing'
                else:
                    self.current_state = 'idle_ground_station'
                self.state_timer = 0
        
        elif self.current_state == 'telemetry_transmission':
            workload = self._simulate_telemetry_transmission()
            if random.random() < 0.28 or self.state_timer > 15:
                self.current_state = 'sensor_fusion_processing'
                self.state_timer = 0
        
        elif self.current_state == 'camera_gimbal_control':
            workload = self._simulate_camera_control()
            if random.random() < 0.32 or self.state_timer > 10:
                self.current_state = 'sensor_fusion_processing'
                self.state_timer = 0
        
        elif self.current_state == 'autonomous_flight':
            workload = self._simulate_autonomous_flight()
            if random.random() < 0.22 or self.state_timer > 25:
                if random.random() < 0.2:
                    self.current_state = 'obstacle_avoidance'
                else:
                    self.current_state = 'flight_stabilization'
                self.state_timer = 0
        
        elif self.current_state == 'emergency_landing':
            workload = self._simulate_emergency_landing()
            if random.random() < 0.15 or self.state_timer > 30:
                self.current_state = 'idle_ground_station'
                self.state_timer = 0
        
        elif self.current_state == 'motor_control':
            workload = self._simulate_motor_control()
            if random.random() < 0.38 or self.state_timer > 5:
                self.current_state = 'flight_stabilization'
                self.state_timer = 0
        
        elif self.current_state == 'waypoint_navigation':
            workload = self._simulate_waypoint_navigation()
            if random.random() < 0.26 or self.state_timer > 22:
                self.current_state = 'gps_navigation'
                self.state_timer = 0
        
        elif self.current_state == 'vision_processing':
            workload = self._simulate_vision_processing()
            if random.random() < 0.33 or self.state_timer > 8:
                self.current_state = 'obstacle_avoidance'
                self.state_timer = 0
        
        # Flight-critical micro-workload spikes (50% of samples - high for real-time control)
        if random.random() < 0.50 and self.current_state != 'idle_ground_station':
            workload += self._generate_flight_workload_spike()
        
        # **PROPER BATTERY MANAGEMENT - WORKING CORRECTLY**
        # Calculate battery level based on progress through simulation
        # Start at 85%, end around 75% after all samples
        progress = self.total_samples_collected / self.samples
        base_battery = 85.0 - (progress * 8.0)  # 8% total drop over simulation
        
        # Add small random variations for realism
        random_variation = random.uniform(-0.3, 0.3)
        
        # Set battery level with bounds
        self.battery_level = max(70.0, min(90.0, base_battery + random_variation))
        
        workload_intensity = min(100, workload / 35)  # Scale to 0-100
        
        return workload, workload_intensity

    def _update_flight_dynamics(self):
        """Simulate realistic flight dynamics and sensor readings"""
        # Altitude variations
        alt_change = random.uniform(-0.5, 0.8)
        self.altitude = max(0.0, self.altitude + alt_change)
        
        # Attitude variations (roll, pitch, yaw)
        self.attitude[0] += random.uniform(-0.3, 0.3)  # roll
        self.attitude[1] += random.uniform(-0.2, 0.2)  # pitch
        self.attitude[2] += random.uniform(-1.0, 1.0)  # yaw
        
        # Keep attitude within reasonable limits
        self.attitude[0] = max(-45.0, min(45.0, self.attitude[0]))
        self.attitude[1] = max(-45.0, min(45.0, self.attitude[1]))
        
        # Motor speed variations
        for i in range(4):
            speed_change = random.randint(-20, 30)
            self.motor_speeds[i] = max(1000, min(2000, self.motor_speeds[i] + speed_change))
        
        # GPS coordinate drift (small changes)
        self.gps_coords[0] += random.uniform(-0.0001, 0.0001)
        self.gps_coords[1] += random.uniform(-0.0001, 0.0001)
        
        # Obstacle distance changes
        self.obstacle_distance += random.uniform(-2.0, 3.0)
        self.obstacle_distance = max(0.0, self.obstacle_distance)
        
        # Vibration level changes
        self.vibration_level = random.uniform(0.01, 0.05)

    def _generate_flight_workload_spike(self):
        """Generate flight-critical micro-workload spikes"""
        workload = 0
        spike_intensity = random.randint(80, 250)
        for i in range(spike_intensity // 12):
            workload += math.sin(i * 0.18) * 32 + math.cos(i * 0.14) * 25
        return workload

    def _simulate_ground_station(self):
        """Simulate ground station communication and monitoring"""
        workload = 20  # Low base for ground operations
        # Communication protocol handling
        for i in range(70):
            workload += math.cos(i * 0.08) * 10
            workload += (i % 20) * 1.2  # Periodic status checks
        
        # System health monitoring
        for i in range(50):
            workload += (i % 12) * 1.8
        
        return workload + random.randint(15, 35)

    def _simulate_sensor_fusion(self):
        """Simulate IMU, GPS, and sensor data fusion"""
        workload = 120  # High workload for sensor fusion
        # Kalman filter processing
        for i in range(280):
            workload += math.sin(i * 0.05) * 45
            workload += (i * 14) % 350 * 0.6
            
            # Sensor correlation and fusion
            if i % 40 == 0:
                workload += 65
        
        # Complementary filter algorithms
        for i in range(180):
            workload += math.tanh(i * 0.025) * 35
            workload += (i % 35) * 2.8
        
        return workload + random.randint(70, 150)

    def _simulate_flight_stabilization(self):
        """Simulate PID control for flight stabilization"""
        workload = 145  # Very high for real-time control
        # PID control loop calculations
        for i in range(320):
            # Proportional term
            workload += abs(self.attitude[0]) * 3.0 + abs(self.attitude[1]) * 3.0
            # Integral term simulation
            workload += math.sin(i * 0.04) * 50
            # Derivative term simulation
            workload += (i * 16) % 400 * 0.7
            
            # Control output calculation
            if i % 30 == 0:
                workload += 80
        
        # Stability augmentation
        for i in range(200):
            workload += (i % 45) * 3.2
        
        return workload + random.randint(85, 170)

    def _simulate_gps_navigation(self):
        """Simulate GPS positioning and navigation algorithms"""
        workload = 95
        # GPS signal processing
        for i in range(240):
            workload += math.cos(i * 0.06) * 35
            workload += (i * 11) % 300 * 0.5
            
            # Position estimation
            if i % 45 == 0:
                workload += 55
        
        # Navigation calculations
        for i in range(150):
            workload += (i % 32) * 3.0
        
        return workload + random.randint(50, 120)

    def _simulate_obstacle_avoidance(self):
        """Simulate obstacle detection and avoidance algorithms"""
        workload = 155  # Critical workload for safety
        # LiDAR/ultrasonic data processing
        for i in range(300):
            workload += (i * 18) % 380  # Distance measurement
            workload += self._sigmoid(i * 0.035) * 70  # Collision probability
            
            # Avoidance maneuver calculation
            if i % 25 == 0:
                workload += 90
        
        # Path planning around obstacles
        for i in range(180):
            workload += math.tanh(i * 0.015) * 45
        
        return workload + random.randint(90, 180)

    def _simulate_battery_management(self):
        """Simulate battery monitoring and power management"""
        workload = 65
        # Battery state estimation
        for i in range(160):
            workload += math.sin(i * 0.07) * 25
            workload += (i * 8) % 220 * 0.4
        
        # Power consumption optimization
        for i in range(100):
            workload += (i % 25) * 2.5
        
        return workload + random.randint(35, 85)

    def _simulate_telemetry_transmission(self):
        """Simulate telemetry data transmission to ground station"""
        workload = 85
        # Data packetization and transmission
        for i in range(200):
            workload += math.cos(i * 0.065) * 30
            workload += (i * 9) % 260 * 0.5
        
        # Communication protocol handling
        for i in range(120):
            workload += (i % 28) * 2.8
        
        return workload + random.randint(45, 100)

    def _simulate_camera_control(self):
        """Simulate camera gimbal stabilization and control"""
        workload = 105
        # Gimbal motor control
        for i in range(220):
            workload += math.sin(i * 0.055) * 38
            workload += (i * 12) % 280 * 0.6
            
            # Image stabilization
            if i % 35 == 0:
                workload += 60
        
        # Camera parameter adjustment
        for i in range(140):
            workload += (i % 30) * 3.2
        
        return workload + random.randint(55, 125)

    def _simulate_autonomous_flight(self):
        """Simulate autonomous flight decision making"""
        workload = 135
        # Autonomous navigation algorithms
        for i in range(290):
            workload += (i * 15) % 360  # Path planning
            workload += math.cos(i * 0.045) * 48  # Decision logic
            
            # Mission planning updates
            if i % 50 == 0:
                workload += 75
        
        # Behavior tree execution
        for i in range(190):
            workload += (i % 40) * 3.4
        
        return workload + random.randint(75, 150)

    def _simulate_emergency_landing(self):
        """Simulate emergency landing procedures"""
        workload = 165  # Highest priority workload
        # Emergency descent control
        for i in range(340):
            workload += (i * 20) % 420  # Rapid control calculations
            workload += self._sigmoid(i * 0.04) * 80  # Safety checks
            
            # Landing site evaluation
            if i % 20 == 0:
                workload += 95
        
        # System integrity monitoring
        for i in range(220):
            workload += math.tanh(i * 0.012) * 55
        
        return workload + random.randint(100, 190)

    def _simulate_motor_control(self):
        """Simulate motor ESC control and RPM management"""
        workload = 125
        # PWM signal generation
        for i in range(260):
            workload += math.sin(i * 0.08) * 42
            workload += (i * 13) % 320 * 0.7
            
            # Motor synchronization
            if i % 15 == 0:
                workload += 70
        
        # ESC communication protocol
        for i in range(160):
            workload += (i % 25) * 3.6
        
        return workload + random.randint(70, 140)

    def _simulate_waypoint_navigation(self):
        """Simulate waypoint following and mission execution"""
        workload = 115
        # Waypoint sequencing
        for i in range(270):
            workload += math.cos(i * 0.052) * 40
            workload += (i * 14) % 330 * 0.6
        
        # Mission progress tracking
        for i in range(170):
            workload += (i % 38) * 3.1
        
        return workload + random.randint(65, 130)

    def _simulate_vision_processing(self):
        """Simulate computer vision for object recognition"""
        workload = 175  # Very high for vision processing
        # Image processing pipeline
        for i in range(350):
            workload += (i * 22) % 450  # Feature extraction
            workload += self._sigmoid(i * 0.025) * 85  # Object classification
            
            # Neural network inference
            if i % 60 == 0:
                workload += 110
        
        # Visual odometry
        for i in range(240):
            workload += math.tanh(i * 0.018) * 60
        
        return workload + random.randint(95, 200)

    def read_hardware_performance_counters(self):
        """Read hardware performance counters for Drone Flight Controller"""
        if not self.working_events:
            return self._generate_realistic_drone_hw_data()
        
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
            return self._generate_realistic_drone_hw_data()

    def _generate_realistic_drone_hw_data(self):
        """Generate realistic Drone Flight Controller hardware counter data"""
        base = time.time_ns() % 1000000
        # Flight controllers have very high real-time requirements
        if self.current_state in ['flight_stabilization', 'emergency_landing', 'motor_control']:
            multiplier = 1.8  # Critical real-time control states
        elif self.current_state in ['obstacle_avoidance', 'vision_processing', 'autonomous_flight']:
            multiplier = 1.6  # High computational states
        elif self.current_state in ['sensor_fusion_processing', 'gps_navigation']:
            multiplier = 1.4  # Medium processing states
        else:
            multiplier = 1.0  # Normal states
        
        return {
            'cpu-cycles': int((2400000 + (base * 41) % 1000000) * multiplier),
            'instructions': int((2100000 + (base * 33) % 900000) * multiplier),
            'branch-instructions': int((160000 + (base * 21) % 80000) * multiplier),
            'branch-misses': int((5000 + (base * 17) % 3500) * multiplier),
            'cache-references': int((220000 + (base * 27) % 110000) * multiplier),
            'cache-misses': int((12000 + (base * 23) % 9000) * multiplier),
            'L1-dcache-loads': int((190000 + (base * 35) % 100000) * multiplier),
            'L1-dcache-load-misses': int((9000 + (base * 15) % 7000) * multiplier),
            'LLC-loads': int((42000 + (base * 9) % 32000) * multiplier),
            'LLC-load-misses': int((2500 + (base * 7) % 1600) * multiplier),
            'stalled-cycles-frontend': int((210000 + (base * 45) % 140000) * multiplier),
            'stalled-cycles-backend': int((180000 + (base * 47) % 120000) * multiplier),
            'bus-cycles': int((80000 + (base * 19) % 40000) * multiplier)
        }

    def calculate_hardware_metrics(self, raw_counters):
        """Calculate hardware performance metrics for Drone Flight Controller"""
        cycles = max(1, raw_counters.get('cpu-cycles', 1))
        instructions = raw_counters.get('instructions', 2000000)
        branches = max(1, raw_counters.get('branch-instructions', 140000))
        branch_misses = raw_counters.get('branch-misses', 4200)
        cache_refs = max(1, raw_counters.get('cache-references', 200000))
        cache_misses = raw_counters.get('cache-misses', 10500)
        l1_loads = max(1, raw_counters.get('L1-dcache-loads', 170000))
        l1_misses = raw_counters.get('L1-dcache-load-misses', 8000)
        llc_loads = max(1, raw_counters.get('LLC-loads', 40000))
        llc_misses = raw_counters.get('LLC-load-misses', 2100)
        frontend_stalls = raw_counters.get('stalled-cycles-frontend', 180000)
        backend_stalls = raw_counters.get('stalled-cycles-backend', 150000)
        
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
        """Get system context with flight parameters"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = float(f.read().strip()) / 1000.0
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
                cpu_freq = float(f.read().strip()) / 1000.0
            return {
                'cpu_temp': cpu_temp, 
                'cpu_freq': cpu_freq,
                'flight_mode': self.flight_mode,
                'signal_strength': self.signal_strength
            }
        except:
            return {
                'cpu_temp': 45.0, 
                'cpu_freq': 1400.0,
                'flight_mode': self.flight_mode,
                'signal_strength': self.signal_strength
            }

    def collect_sample(self):
        """Collect one comprehensive drone flight controller sample"""
        timestamp = datetime.now()
        
        # Simulate Drone Flight Controller workload
        workload_raw, workload_intensity = self.simulate_drone_workload()
        
        # Read hardware counters
        hw_counters = self.read_hardware_performance_counters()
        
        # Calculate hardware metrics
        hw_metrics = self.calculate_hardware_metrics(hw_counters)
        
        # Get system context
        system_info = self.get_system_context()
        
        # Drone Flight Controller specific metrics
        drone_metrics = {
            'altitude_m': self.altitude,
            'battery_level': self.battery_level,
            'gps_satellites': self.gps_satellites,
            'motor_speed_avg': sum(self.motor_speeds) / len(self.motor_speeds),
            'roll_angle': self.attitude[0],
            'pitch_angle': self.attitude[1],
            'yaw_angle': self.attitude[2],
            'obstacle_distance_m': self.obstacle_distance,
            'airspeed_ms': self.airspeed,
            'vibration_level': self.vibration_level
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
        sample.update(drone_metrics)
        
        return sample

    def run_collection(self):
        """Main collection loop for Drone Flight Controller"""
        print("=== DRONE FLIGHT CONTROLLER HARDWARE PERFORMANCE COUNTERS ===")
        print("Simulating Drone Flight Controller on Raspberry Pi")
        print(f"Target: {self.samples} samples in {self.duration} seconds")
        print(f"Hardware counters available: {len(self.working_events)}")
        print("-" * 60)
        
        start_time = time.time()
        stabilization_count = 0
        sensor_fusion_count = 0
        
        for i in range(self.samples):
            sample_start = time.time()
            
            sample = self.collect_sample()
            self.data.append(sample)
            
            # Count flight control events
            if sample['workload_state'] == 2:  # Flight stabilization
                stabilization_count += 1
            if sample['workload_state'] == 1:  # Sensor fusion
                sensor_fusion_count += 1
            
            if i % 500 == 0:
                elapsed = time.time() - start_time
                progress = (i / self.samples) * 100
                stabilization_percent = (stabilization_count / (i + 1)) * 100
                sensor_fusion_percent = (sensor_fusion_count / (i + 1)) * 100
                
                print(f"Sample {i:5d}/{self.samples} ({progress:5.1f}%)")
                print(f"  State: {self.current_state:25} | "
                      f"Intensity: {sample['workload_intensity']:5.1f}")
                print(f"  Stabilization: {stabilization_percent:5.1f}% | "
                      f"Sensor Fusion: {sensor_fusion_percent:5.1f}% | "
                      f"Mode: {sample['flight_mode']:10s}")
                print(f"  Altitude: {sample['altitude_m']:5.1f}m | "
                      f"Battery: {sample['battery_level']:5.1f}% | "
                      f"Motors: {sample['motor_speed_avg']:4.0f}RPM")
                print(f"  Roll: {sample['roll_angle']:5.1f}° | "
                      f"Pitch: {sample['pitch_angle']:5.1f}° | "
                      f"Obstacle: {sample['obstacle_distance_m']:5.1f}m")
                print(f"  IPC: {sample['hw_instructions_per_cycle']:5.3f} | "
                      f"Signal: {sample['signal_strength']:3d} dBm")
                print()
            
            elapsed_sample = time.time() - sample_start
            time.sleep(max(0.001, self.interval - elapsed_sample))
        
        self.save_data()

    def save_data(self):
        """Save collected drone flight controller data to CSV"""
        filename = f"drone_flight_controller_hpc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
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
        stabilization_samples = sum(1 for s in self.data if s['workload_state'] == 2)
        sensor_fusion_samples = sum(1 for s in self.data if s['workload_state'] == 1)
        stabilization_percent = (stabilization_samples / total_samples) * 100
        sensor_fusion_percent = (sensor_fusion_samples / total_samples) * 100
        avg_intensity = sum(s['workload_intensity'] for s in self.data) / total_samples
        avg_ipc = sum(s['hw_instructions_per_cycle'] for s in self.data) / total_samples
        
        print(f"Saved {total_samples} samples to {filename}")
        print(f"Flight stabilization samples: {stabilization_samples}/{total_samples} ({stabilization_percent:.1f}%)")
        print(f"Sensor fusion samples: {sensor_fusion_samples}/{total_samples} ({sensor_fusion_percent:.1f}%)")
        print(f"Average workload intensity: {avg_intensity:.1f}")
        print(f"Average Instructions Per Cycle: {avg_ipc:.3f}")
        
        # Flight specific statistics
        avg_altitude = sum(s['altitude_m'] for s in self.data) / total_samples
        avg_battery = sum(s['battery_level'] for s in self.data) / total_samples
        print(f"Average Altitude: {avg_altitude:.1f} meters")
        print(f"Average Battery Level: {avg_battery:.1f}%")

if __name__ == "__main__":
    # Enable performance monitoring
    os.system("echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid > /dev/null 2>&1")
    collector = DroneFlightControllerHPC(samples=20000, duration=600)
    collector.run_collection()
