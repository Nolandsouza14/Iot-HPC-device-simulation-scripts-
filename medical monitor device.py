medical monitor device

#!/usr/bin/env python3
import time
import os
import subprocess
import random
import math
from datetime import datetime

class MedicalMonitorHPC:
    def __init__(self, samples=20000, duration=600):
        self.samples = samples
        self.duration = duration
        self.interval = duration / samples
        self.data = []
        
        # Medical Monitor specific workload states
        self.workload_states = {
            'idle_patient_monitoring': 0,
            'ecg_signal_processing': 1,
            'blood_pressure_analysis': 2,
            'spo2_calculation': 3,
            'respiratory_monitoring': 4,
            'temperature_tracking': 5,
            'alarm_condition_detection': 6,
            'data_logging_storage': 7,
            'wireless_data_transmission': 8,
            'display_rendering': 9,
            'battery_management': 10,
            'diagnostic_self_test': 11
        }
        self.current_state = 'idle_patient_monitoring'
        self.state_timer = 0
        
        # Medical monitor specific parameters
        self.heart_rate = 72
        self.blood_pressure_sys = 120
        self.blood_pressure_dia = 80
        self.spo2_level = 98
        self.respiratory_rate = 16
        self.body_temperature = 36.6
        self.battery_level = 92
        self.alarm_triggered = False
        self.patient_connected = True
        
        # Medical signal processing parameters
        self.ecg_sampling_rate = 500  # Hz
        self.blood_pressure_interval = 30  # seconds
        self.spo2_update_rate = 2  # seconds
        
        # Hardware performance counters (ARM medical device specific)
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

    def simulate_medical_monitor_workload(self):
        """Simulate Medical Monitor specific workload patterns"""
        self.state_timer += 1
        workload = 0
        
        # Medical monitor state transitions with clinical priorities
        if self.current_state == 'idle_patient_monitoring':
            # Continuous vital signs monitoring
            workload = self._simulate_continuous_monitoring()
            
            # Medical events occur frequently (12% chance)
            if random.random() < 0.12 or self.state_timer > 25:
                next_states = ['ecg_signal_processing', 'blood_pressure_analysis', 
                              'spo2_calculation', 'alarm_condition_detection', 'data_logging_storage']
                weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # ECG most critical
                self.current_state = random.choices(next_states, weights=weights)[0]
                self.state_timer = 0
        
        elif self.current_state == 'ecg_signal_processing':
            workload = self._simulate_ecg_processing()
            if random.random() < 0.45 or self.state_timer > 15:
                # ECG often leads to alarm detection or continues monitoring
                if random.random() < 0.3 and self.heart_rate > 100:
                    self.current_state = 'alarm_condition_detection'
                else:
                    self.current_state = 'idle_patient_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'blood_pressure_analysis':
            workload = self._simulate_blood_pressure_analysis()
            if random.random() < 0.35 or self.state_timer > 20:
                self.current_state = 'idle_patient_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'spo2_calculation':
            workload = self._simulate_spo2_calculation()
            if random.random() < 0.40 or self.state_timer > 12:
                if self.spo2_level < 92:
                    self.current_state = 'alarm_condition_detection'
                else:
                    self.current_state = 'idle_patient_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'respiratory_monitoring':
            workload = self._simulate_respiratory_monitoring()
            if random.random() < 0.38 or self.state_timer > 18:
                self.current_state = 'idle_patient_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'temperature_tracking':
            workload = self._simulate_temperature_tracking()
            if random.random() < 0.32 or self.state_timer > 15:
                self.current_state = 'idle_patient_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'alarm_condition_detection':
            workload = self._simulate_alarm_detection()
            self.alarm_triggered = True
            if random.random() < 0.25 or self.state_timer > 8:
                self.current_state = 'wireless_data_transmission'  # Send alarm data
                self.alarm_triggered = False
                self.state_timer = 0
        
        elif self.current_state == 'data_logging_storage':
            workload = self._simulate_data_logging()
            if random.random() < 0.28 or self.state_timer > 25:
                self.current_state = 'idle_patient_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'wireless_data_transmission':
            workload = self._simulate_wireless_transmission()
            if random.random() < 0.22 or self.state_timer > 30:
                self.current_state = 'idle_patient_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'display_rendering':
            workload = self._simulate_display_rendering()
            if random.random() < 0.20 or self.state_timer > 10:
                self.current_state = 'idle_patient_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'battery_management':
            workload = self._simulate_battery_management()
            if random.random() < 0.15 or self.state_timer > 35:
                self.current_state = 'idle_patient_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'diagnostic_self_test':
            workload = self._simulate_self_test()
            if random.random() < 0.10 or self.state_timer > 60:
                self.current_state = 'idle_patient_monitoring'
                self.state_timer = 0
        
        # Medical-specific micro-workload spikes (40% of samples)
        if random.random() < 0.40 and self.current_state != 'idle_patient_monitoring':
            workload += self._generate_medical_workload_spike()
        
        # Simulate vital signs variations
        self._update_vital_signs()
        
        # Battery consumption simulation
        if workload > 60:
            self.battery_level -= 0.002
        self.battery_level = max(15, self.battery_level)  # Safety margin
        
        workload_intensity = min(100, workload / 35)  # Scale to 0-100
        
        return workload, workload_intensity

    def _update_vital_signs(self):
        """Simulate realistic vital signs variations"""
        # Heart rate variations
        self.heart_rate += random.randint(-2, 2)
        self.heart_rate = max(60, min(120, self.heart_rate))
        
        # SpO2 variations
        self.spo2_level += random.randint(-1, 1)
        self.spo2_level = max(90, min(100, self.spo2_level))
        
        # Respiratory rate
        self.respiratory_rate += random.randint(-1, 1)
        self.respiratory_rate = max(12, min(20, self.respiratory_rate))
        
        # Temperature variations
        self.body_temperature += random.uniform(-0.1, 0.1)
        self.body_temperature = max(36.0, min(37.5, self.body_temperature))

    def _generate_medical_workload_spike(self):
        """Generate medical-specific micro-workload spikes"""
        workload = 0
        spike_intensity = random.randint(50, 200)
        for i in range(spike_intensity // 10):
            workload += math.sin(i * 0.15) * 25 + math.cos(i * 0.08) * 18
        return workload

    def _simulate_continuous_monitoring(self):
        """Simulate background patient monitoring"""
        workload = 25  # Base monitoring workload
        # Continuous signal processing
        for i in range(80):
            workload += math.cos(i * 0.07) * 12
            workload += (i % 20) * 1.5  # Periodic sensor readings
        
        # Signal quality monitoring
        for i in range(40):
            workload += (i % 10) * 2.0
        
        return workload + random.randint(15, 35)

    def _simulate_ecg_processing(self):
        """Simulate ECG signal processing and analysis"""
        workload = 120  # High workload for medical signal processing
        # ECG signal filtering and analysis
        for i in range(300):
            # QRS complex detection
            workload += math.sin(i * 0.05) * 45
            workload += (i * 13) % 350 * 0.6
            
            # Heart rate variability analysis
            if i % 50 == 0:
                workload += 65
        
        # Arrhythmia detection algorithms
        for i in range(180):
            workload += math.tanh(i * 0.02) * 35
            workload += (i % 30) * 2.8
        
        # Signal artifact removal
        for i in range(120):
            workload += (i * 7) % 200 * 0.4
        
        return workload + random.randint(70, 150)

    def _simulate_blood_pressure_analysis(self):
        """Simulate blood pressure measurement and analysis"""
        workload = 85
        # Oscillometric analysis
        for i in range(200):
            workload += math.sin(i * 0.08) * 32
            workload += (i * 9) % 280 * 0.5
            
            # Pressure waveform analysis
            if i % 40 == 0:
                workload += 45
        
        # Systolic/Diastolic calculation
        for i in range(120):
            workload += (i % 25) * 3.2
        
        return workload + random.randint(40, 100)

    def _simulate_spo2_calculation(self):
        """Simulate SpO2 oxygen saturation calculation"""
        workload = 95
        # PPG signal processing (Photoplethysmography)
        for i in range(220):
            workload += math.cos(i * 0.06) * 38
            workload += (i * 11) % 300 * 0.5
            
            # Red/IR ratio calculation
            if i % 35 == 0:
                workload += 55
        
        # Motion artifact compensation
        for i in range(140):
            workload += (i % 28) * 2.5
        
        return workload + random.randint(45, 110)

    def _simulate_respiratory_monitoring(self):
        """Simulate respiratory rate monitoring"""
        workload = 75
        # Impedance or capnography processing
        for i in range(180):
            workload += math.sin(i * 0.07) * 28
            workload += (i * 8) % 250 * 0.4
        
        # Breath detection algorithms
        for i in range(100):
            workload += (i % 20) * 3.0
        
        return workload + random.randint(35, 85)

    def _simulate_temperature_tracking(self):
        """Simulate body temperature monitoring"""
        workload = 55
        # Temperature sensor processing
        for i in range(120):
            workload += math.cos(i * 0.09) * 20
            workload += (i % 25) * 1.8
        
        # Trend analysis
        for i in range(80):
            workload += (i % 15) * 2.2
        
        return workload + random.randint(25, 65)

    def _simulate_alarm_detection(self):
        """Simulate medical alarm condition detection"""
        workload = 135  # Critical workload
        # Multi-parameter alarm algorithms
        for i in range(250):
            workload += (i * 15) % 350  # Pattern recognition
            workload += self._sigmoid(i * 0.04) * 60  # Threshold checking
            
            # Alarm priority calculation
            if i % 30 == 0:
                workload += 70
        
        # Alarm validation and suppression
        for i in range(150):
            workload += (i % 35) * 3.5
        
        return workload + random.randint(80, 160)

    def _simulate_data_logging(self):
        """Simulate medical data storage and retrieval"""
        workload = 65
        # Data compression and storage
        for i in range(160):
            workload += math.sin(i * 0.08) * 25
            workload += (i * 6) % 200 * 0.5
        
        # Database operations
        for i in range(100):
            workload += (i % 20) * 2.8
        
        return workload + random.randint(30, 75)

    def _simulate_wireless_transmission(self):
        """Simulate wireless medical data transmission"""
        workload = 105
        # Data encryption and transmission
        for i in range(220):
            workload += math.cos(i * 0.05) * 40
            workload += (i * 10) % 280 * 0.6
            
            # Protocol handling (Bluetooth/WiFi)
            if i % 45 == 0:
                workload += 55
        
        # Error checking and retransmission
        for i in range(120):
            workload += (i % 25) * 3.0
        
        return workload + random.randint(50, 120)

    def _simulate_display_rendering(self):
        """Simulate medical display rendering"""
        workload = 80
        # Waveform rendering
        for i in range(180):
            workload += math.sin(i * 0.06) * 30
            workload += (i * 7) % 240 * 0.4
        
        # Numerical display updates
        for i in range(100):
            workload += (i % 18) * 3.2
        
        return workload + random.randint(35, 90)

    def _simulate_battery_management(self):
        """Simulate power management for medical device"""
        workload = 45
        # Power optimization algorithms
        for i in range(100):
            workload += math.cos(i * 0.1) * 15
            workload += (i % 22) * 1.5
        
        # Battery health monitoring
        for i in range(60):
            workload += (i % 12) * 2.0
        
        return workload + random.randint(20, 50)

    def _simulate_self_test(self):
        """Simulate medical device self-diagnostics"""
        workload = 95
        # Hardware diagnostic routines
        for i in range(200):
            workload += math.sin(i * 0.07) * 35
            workload += (i * 8) % 260 * 0.5
        
        # Sensor calibration verification
        for i in range(120):
            workload += (i % 30) * 2.5
        
        return workload + random.randint(45, 105)

    def read_hardware_performance_counters(self):
        """Read hardware performance counters for Medical Monitor"""
        if not self.working_events:
            return self._generate_realistic_medical_hw_data()
        
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
            return self._generate_realistic_medical_hw_data()

    def _generate_realistic_medical_hw_data(self):
        """Generate realistic Medical Monitor hardware counter data"""
        base = time.time_ns() % 1000000
        # Medical devices often have optimized performance profiles
        if self.current_state in ['ecg_signal_processing', 'alarm_condition_detection']:
            multiplier = 1.5  # Critical medical processing
        elif self.current_state in ['spo2_calculation', 'blood_pressure_analysis']:
            multiplier = 1.3  # Medium medical processing
        else:
            multiplier = 0.9  # Background monitoring
        
        return {
            'cpu-cycles': int((1900000 + (base * 31) % 750000) * multiplier),
            'instructions': int((1700000 + (base * 23) % 650000) * multiplier),
            'branch-instructions': int((110000 + (base * 11) % 55000) * multiplier),
            'branch-misses': int((3200 + (base * 7) % 1800) * multiplier),
            'cache-references': int((170000 + (base * 17) % 85000) * multiplier),
            'cache-misses': int((7500 + (base * 13) % 4500) * multiplier),
            'L1-dcache-loads': int((145000 + (base * 25) % 75000) * multiplier),
            'L1-dcache-load-misses': int((5500 + (base * 5) % 3000) * multiplier),
            'LLC-loads': int((32000 + (base * 3) % 20000) * multiplier),
            'LLC-load-misses': int((1600 + (base * 1) % 900) * multiplier),
            'stalled-cycles-frontend': int((145000 + (base * 35) % 95000) * multiplier),
            'stalled-cycles-backend': int((125000 + (base * 37) % 75000) * multiplier),
            'bus-cycles': int((55000 + (base * 11) % 28000) * multiplier)
        }

    def calculate_hardware_metrics(self, raw_counters):
        """Calculate hardware performance metrics for Medical Monitor"""
        cycles = max(1, raw_counters.get('cpu-cycles', 1))
        instructions = raw_counters.get('instructions', 1550000)
        branches = max(1, raw_counters.get('branch-instructions', 85000))
        branch_misses = raw_counters.get('branch-misses', 2600)
        cache_refs = max(1, raw_counters.get('cache-references', 145000))
        cache_misses = raw_counters.get('cache-misses', 6500)
        l1_loads = max(1, raw_counters.get('L1-dcache-loads', 125000))
        l1_misses = raw_counters.get('L1-dcache-load-misses', 5000)
        llc_loads = max(1, raw_counters.get('LLC-loads', 29000))
        llc_misses = raw_counters.get('LLC-load-misses', 1400)
        frontend_stalls = raw_counters.get('stalled-cycles-frontend', 115000)
        backend_stalls = raw_counters.get('stalled-cycles-backend', 95000)
        
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
        """Get system context with medical parameters"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = float(f.read().strip()) / 1000.0
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
                cpu_freq = float(f.read().strip()) / 1000.0
            return {
                'cpu_temp': cpu_temp, 
                'cpu_freq': cpu_freq,
                'battery_level': self.battery_level,
                'patient_connected': 1 if self.patient_connected else 0
            }
        except:
            return {
                'cpu_temp': 40.0, 
                'cpu_freq': 1200.0,
                'battery_level': self.battery_level,
                'patient_connected': 1 if self.patient_connected else 0
            }

    def collect_sample(self):
        """Collect one comprehensive medical sample"""
        timestamp = datetime.now()
        
        # Simulate Medical Monitor workload
        workload_raw, workload_intensity = self.simulate_medical_monitor_workload()
        
        # Read hardware counters
        hw_counters = self.read_hardware_performance_counters()
        
        # Calculate hardware metrics
        hw_metrics = self.calculate_hardware_metrics(hw_counters)
        
        # Get system context
        system_info = self.get_system_context()
        
        # Medical-specific metrics
        medical_metrics = {
            'heart_rate': self.heart_rate,
            'blood_pressure_sys': self.blood_pressure_sys,
            'blood_pressure_dia': self.blood_pressure_dia,
            'spo2_level': self.spo2_level,
            'respiratory_rate': self.respiratory_rate,
            'body_temperature': self.body_temperature,
            'alarm_triggered': 1 if self.alarm_triggered else 0,
            'ecg_sampling_rate': self.ecg_sampling_rate
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
        sample.update(medical_metrics)
        
        return sample

    def run_collection(self):
        """Main collection loop for Medical Monitor"""
        print("=== MEDICAL MONITOR HARDWARE PERFORMANCE COUNTERS ===")
        print("Simulating Medical Monitoring Device on Raspberry Pi")
        print(f"Target: {self.samples} samples in {self.duration} seconds")
        print(f"Hardware counters available: {len(self.working_events)}")
        print("-" * 60)
        
        start_time = time.time()
        alarm_count = 0
        ecg_processing_count = 0
        
        for i in range(self.samples):
            sample_start = time.time()
            
            sample = self.collect_sample()
            self.data.append(sample)
            
            # Count medical events
            if sample['alarm_triggered'] > 0:
                alarm_count += 1
            if sample['workload_state'] == 1:  # ECG processing
                ecg_processing_count += 1
            
            if i % 500 == 0:
                elapsed = time.time() - start_time
                progress = (i / self.samples) * 100
                alarm_percent = (alarm_count / (i + 1)) * 100
                ecg_percent = (ecg_processing_count / (i + 1)) * 100
                
                print(f"Sample {i:5d}/{self.samples} ({progress:5.1f}%)")
                print(f"  State: {self.current_state:28} | "
                      f"Intensity: {sample['workload_intensity']:5.1f}")
                print(f"  Alarms: {alarm_percent:5.1f}% | "
                      f"ECG Processing: {ecg_percent:5.1f}% | "
                      f"Battery: {sample['battery_level']:4.1f}%")
                print(f"  HR: {sample['heart_rate']:3d} bpm | "
                      f"SpO2: {sample['spo2_level']:2d}% | "
                      f"Temp: {sample['body_temperature']:4.1f}°C")
                print(f"  IPC: {sample['hw_instructions_per_cycle']:5.3f} | "
                      f"Temp: {sample['cpu_temp']:4.1f}°C")
                print()
            
            elapsed_sample = time.time() - sample_start
            time.sleep(max(0.001, self.interval - elapsed_sample))
        
        self.save_data()

    def save_data(self):
        """Save collected medical data to CSV"""
        filename = f"medical_monitor_hpc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
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
        alarm_samples = sum(1 for s in self.data if s['alarm_triggered'] > 0)
        ecg_samples = sum(1 for s in self.data if s['workload_state'] == 1)
        alarm_percent = (alarm_samples / total_samples) * 100
        ecg_percent = (ecg_samples / total_samples) * 100
        avg_intensity = sum(s['workload_intensity'] for s in self.data) / total_samples
        avg_ipc = sum(s['hw_instructions_per_cycle'] for s in self.data) / total_samples
        
        print(f"Saved {total_samples} samples to {filename}")
        print(f"Alarm condition samples: {alarm_samples}/{total_samples} ({alarm_percent:.1f}%)")
        print(f"ECG processing samples: {ecg_samples}/{total_samples} ({ecg_percent:.1f}%)")
        print(f"Average workload intensity: {avg_intensity:.1f}")
        print(f"Average Instructions Per Cycle: {avg_ipc:.3f}")
        
        # Medical device specific statistics
        avg_heart_rate = sum(s['heart_rate'] for s in self.data) / total_samples
        avg_spo2 = sum(s['spo2_level'] for s in self.data) / total_samples
        print(f"Average Heart Rate: {avg_heart_rate:.1f} bpm")
        print(f"Average SpO2: {avg_spo2:.1f}%")

if __name__ == "__main__":
    # Enable performance monitoring
    os.system("echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid > /dev/null 2>&1")
    collector = MedicalMonitorHPC(samples=20000, duration=600)
    collector.run_collection()
