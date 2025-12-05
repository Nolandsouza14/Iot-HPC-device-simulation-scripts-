tplink omada

#!/usr/bin/env python3
import time
import os
import subprocess
import random
import math
from datetime import datetime

class TPLinkOmadaHPC:
    def __init__(self, samples=20000, duration=600):
        self.samples = samples
        self.duration = duration
        self.interval = duration / samples
        self.data = []
        
        # TP-Link Omada Controller specific workload states
        self.workload_states = {
            'idle_network_monitoring': 0,
            'client_device_management': 1,
            'wireless_ap_control': 2,
            'network_topology_discovery': 3,
            'qos_policy_enforcement': 4,
            'security_monitoring': 5,
            'bandwidth_optimization': 6,
            'roaming_optimization': 7,
            'firmware_management': 8,
            'cloud_sync_processing': 9,
            'vlan_configuration': 10,
            'traffic_analysis': 11,
            'mesh_network_optimization': 12
        }
        self.current_state = 'idle_network_monitoring'
        self.state_timer = 0
        
        # Omada Controller specific parameters
        self.connected_devices = 45
        self.managed_aps = 8
        self.network_throughput = 750  # Mbps
        self.cpu_utilization = 25
        self.memory_usage = 380  # MB
        self.security_events = 2
        self.connected_clients = 62
        
        # Network performance metrics
        self.packet_loss_rate = 0.02  # %
        self.network_latency = 12  # ms
        self.wifi_interference = 3
        
        # Hardware performance counters (ARM network controller specific)
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

    def simulate_omada_workload(self):
        """Simulate TP-Link Omada Controller specific workload patterns"""
        self.state_timer += 1
        workload = 0
        
        # Update network conditions
        self._update_network_conditions()
        
        # Omada Controller state transitions with network priorities
        if self.current_state == 'idle_network_monitoring':
            # Continuous network monitoring
            workload = self._simulate_network_monitoring()
            
            # Network events occur frequently (18% chance - high for network controller)
            if random.random() < 0.18 or self.state_timer > 15:
                next_states = ['client_device_management', 'wireless_ap_control', 
                              'security_monitoring', 'traffic_analysis', 'qos_policy_enforcement']
                weights = [0.25, 0.2, 0.15, 0.2, 0.2]
                self.current_state = random.choices(next_states, weights=weights)[0]
                self.state_timer = 0
        
        elif self.current_state == 'client_device_management':
            workload = self._simulate_client_management()
            if random.random() < 0.35 or self.state_timer > 12:
                # Client management often leads to traffic analysis
                if random.random() < 0.4:
                    self.current_state = 'traffic_analysis'
                else:
                    self.current_state = 'idle_network_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'wireless_ap_control':
            workload = self._simulate_ap_control()
            if random.random() < 0.30 or self.state_timer > 18:
                # AP control may trigger roaming optimization
                if random.random() < 0.3:
                    self.current_state = 'roaming_optimization'
                else:
                    self.current_state = 'idle_network_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'network_topology_discovery':
            workload = self._simulate_topology_discovery()
            if random.random() < 0.25 or self.state_timer > 25:
                self.current_state = 'idle_network_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'qos_policy_enforcement':
            workload = self._simulate_qos_enforcement()
            if random.random() < 0.40 or self.state_timer > 10:
                self.current_state = 'bandwidth_optimization'
                self.state_timer = 0
        
        elif self.current_state == 'security_monitoring':
            workload = self._simulate_security_monitoring()
            if random.random() < 0.28 or self.state_timer > 15:
                if self.security_events > 5:
                    self.current_state = 'client_device_management'  # Block devices
                else:
                    self.current_state = 'idle_network_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'bandwidth_optimization':
            workload = self._simulate_bandwidth_optimization()
            if random.random() < 0.32 or self.state_timer > 20:
                self.current_state = 'idle_network_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'roaming_optimization':
            workload = self._simulate_roaming_optimization()
            if random.random() < 0.22 or self.state_timer > 22:
                self.current_state = 'idle_network_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'firmware_management':
            workload = self._simulate_firmware_management()
            if random.random() < 0.15 or self.state_timer > 40:
                self.current_state = 'idle_network_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'cloud_sync_processing':
            workload = self._simulate_cloud_sync()
            if random.random() < 0.20 or self.state_timer > 30:
                self.current_state = 'idle_network_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'vlan_configuration':
            workload = self._simulate_vlan_configuration()
            if random.random() < 0.18 or self.state_timer > 28:
                self.current_state = 'idle_network_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'traffic_analysis':
            workload = self._simulate_traffic_analysis()
            if random.random() < 0.38 or self.state_timer > 16:
                self.current_state = 'idle_network_monitoring'
                self.state_timer = 0
        
        elif self.current_state == 'mesh_network_optimization':
            workload = self._simulate_mesh_optimization()
            if random.random() < 0.25 or self.state_timer > 35:
                self.current_state = 'idle_network_monitoring'
                self.state_timer = 0
        
        # Network-specific micro-workload spikes (45% of samples - high for networking)
        if random.random() < 0.45 and self.current_state != 'idle_network_monitoring':
            workload += self._generate_network_workload_spike()
        
        # Calculate workload intensity FIRST
        workload_intensity = min(100, workload / 40)  # Scale to 0-100
        
        # THEN update system metrics based on workload_intensity
        self.cpu_utilization = min(95, 20 + (workload_intensity * 0.8))
        self.memory_usage = min(512, 350 + (workload_intensity * 1.5))
        
        return workload, workload_intensity

    def _update_network_conditions(self):
        """Simulate realistic network condition changes"""
        # Connected clients fluctuation
        client_change = random.randint(-3, 5)
        self.connected_clients = max(10, min(150, self.connected_clients + client_change))
        
        # Network throughput variations
        throughput_change = random.randint(-50, 100)
        self.network_throughput = max(100, min(1000, self.network_throughput + throughput_change))
        
        # Security events (occasional)
        if random.random() < 0.05:
            self.security_events += random.randint(1, 3)
        else:
            self.security_events = max(0, self.security_events - random.randint(0, 1))
        
        # Packet loss and interference
        self.packet_loss_rate = random.uniform(0.01, 0.1)
        self.wifi_interference = random.randint(1, 5)

    def _generate_network_workload_spike(self):
        """Generate network-specific micro-workload spikes"""
        workload = 0
        spike_intensity = random.randint(60, 220)
        for i in range(spike_intensity // 10):
            workload += math.sin(i * 0.16) * 28 + math.cos(i * 0.11) * 22
        return workload

    def _simulate_network_monitoring(self):
        """Simulate continuous network monitoring"""
        workload = 30  # Moderate base for network monitoring
        # Continuous packet inspection and monitoring
        for i in range(100):
            workload += math.cos(i * 0.06) * 15
            workload += (i % 25) * 1.8  # Periodic health checks
        
        # Device status monitoring
        for i in range(60):
            workload += (i % 15) * 2.2
        
        return workload + random.randint(20, 40)

    def _simulate_client_management(self):
        """Simulate client device management and authentication"""
        workload = 95
        # Client authentication and session management
        for i in range(220):
            workload += math.sin(i * 0.07) * 35
            workload += (i * 11) % 280 * 0.5
            
            # DHCP and IP management
            if i % 40 == 0:
                workload += 50
        
        # Client policy enforcement
        for i in range(140):
            workload += (i % 30) * 3.0
        
        return workload + random.randint(45, 110)

    def _simulate_ap_control(self):
        """Simulate wireless access point management"""
        workload = 110
        # AP configuration and monitoring
        for i in range(260):
            workload += (i * 13) % 320  # Radio management
            workload += math.cos(i * 0.05) * 45  # Channel optimization
            
            # Transmit power adjustments
            if i % 45 == 0:
                workload += 60
        
        # AP health monitoring
        for i in range(160):
            workload += (i % 35) * 3.2
        
        return workload + random.randint(55, 130)

    def _simulate_topology_discovery(self):
        """Simulate network topology discovery and mapping"""
        workload = 125
        # Network discovery algorithms
        for i in range(280):
            workload += math.sin(i * 0.04) * 48
            workload += (i * 15) % 350 * 0.6
            
            # Device relationship mapping
            if i % 50 == 0:
                workload += 70
        
        # Path calculation and optimization
        for i in range(180):
            workload += (i % 40) * 3.5
        
        return workload + random.randint(65, 140)

    def _simulate_qos_enforcement(self):
        """Simulate Quality of Service policy enforcement"""
        workload = 105
        # Traffic classification and prioritization
        for i in range(240):
            workload += math.cos(i * 0.06) * 38
            workload += (i * 12) % 300 * 0.5
            
            # Bandwidth allocation algorithms
            if i % 35 == 0:
                workload += 55
        
        # Policy rule enforcement
        for i in range(150):
            workload += (i % 32) * 3.0
        
        return workload + random.randint(50, 120)

    def _simulate_security_monitoring(self):
        """Simulate network security monitoring and threat detection"""
        workload = 135
        # Intrusion detection system processing
        for i in range(300):
            workload += (i * 16) % 380  # Pattern matching
            workload += self._sigmoid(i * 0.03) * 60  # Threat scoring
            
            # Security rule evaluation
            if i % 40 == 0:
                workload += 75
        
        # Log analysis and correlation
        for i in range(200):
            workload += math.tanh(i * 0.02) * 40
        
        return workload + random.randint(70, 150)

    def _simulate_bandwidth_optimization(self):
        """Simulate bandwidth management and optimization"""
        workload = 115
        # Traffic shaping algorithms
        for i in range(250):
            workload += math.sin(i * 0.05) * 42
            workload += (i * 14) % 320 * 0.5
            
            # Congestion control
            if i % 45 == 0:
                workload += 65
        
        # Load balancing calculations
        for i in range(170):
            workload += (i % 38) * 3.3
        
        return workload + random.randint(60, 130)

    def _simulate_roaming_optimization(self):
        """Simulate client roaming optimization between APs"""
        workload = 120
        # Roaming decision algorithms
        for i in range(270):
            workload += math.cos(i * 0.055) * 44
            workload += (i * 13) % 330 * 0.6
            
            # Signal strength analysis
            if i % 42 == 0:
                workload += 68
        
        # Seamless handover management
        for i in range(160):
            workload += (i % 36) * 3.4
        
        return workload + random.randint(65, 135)

    def _simulate_firmware_management(self):
        """Simulate AP firmware management and updates"""
        workload = 140
        # Firmware distribution and verification
        for i in range(290):
            workload += (i * 17) % 360  # File transfer management
            workload += math.sin(i * 0.045) * 50  # Update scheduling
            
            # Version compatibility checks
            if i % 55 == 0:
                workload += 80
        
        # Rollback and recovery procedures
        for i in range(190):
            workload += (i % 42) * 3.6
        
        return workload + random.randint(75, 145)

    def _simulate_cloud_sync(self):
        """Simulate cloud configuration synchronization"""
        workload = 90
        # Cloud communication and data sync
        for i in range(200):
            workload += math.cos(i * 0.07) * 32
            workload += (i * 10) % 260 * 0.5
        
        # Conflict resolution and merge
        for i in range(130):
            workload += (i % 28) * 2.8
        
        return workload + random.randint(45, 100)

    def _simulate_vlan_configuration(self):
        """Simulate VLAN configuration and management"""
        workload = 100
        # VLAN setup and tagging
        for i in range(230):
            workload += math.sin(i * 0.065) * 36
            workload += (i * 11) % 290 * 0.5
        
        # Security policy application
        for i in range(140):
            workload += (i % 34) * 2.9
        
        return workload + random.randint(50, 110)

    def _simulate_traffic_analysis(self):
        """Simulate network traffic analysis and reporting"""
        workload = 130
        # Deep packet inspection
        for i in range(280):
            workload += (i * 15) % 350  # Protocol analysis
            workload += math.cos(i * 0.048) * 46  # Traffic pattern recognition
            
            # Statistical analysis
            if i % 48 == 0:
                workload += 72
        
        # Report generation
        for i in range(180):
            workload += (i % 40) * 3.1
        
        return workload + random.randint(70, 140)

    def _simulate_mesh_optimization(self):
        """Simulate mesh network optimization"""
        workload = 145
        # Mesh path optimization algorithms
        for i in range(310):
            workload += math.sin(i * 0.042) * 52
            workload += (i * 18) % 380 * 0.6
            
            # Backhaul optimization
            if i % 52 == 0:
                workload += 85
        
        # Self-healing network algorithms
        for i in range(200):
            workload += math.tanh(i * 0.018) * 45
        
        return workload + random.randint(80, 155)

    def read_hardware_performance_counters(self):
        """Read hardware performance counters for Omada Controller"""
        if not self.working_events:
            return self._generate_realistic_omada_hw_data()
        
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
            return self._generate_realistic_omada_hw_data()

    def _generate_realistic_omada_hw_data(self):
        """Generate realistic Omada Controller hardware counter data"""
        base = time.time_ns() % 1000000
        # Network controllers have higher baseline activity
        if self.current_state in ['security_monitoring', 'traffic_analysis', 'mesh_network_optimization']:
            multiplier = 1.6  # High network processing states
        elif self.current_state in ['qos_policy_enforcement', 'topology_discovery', 'ap_control']:
            multiplier = 1.4  # Medium network control states
        else:
            multiplier = 1.1  # Network monitoring has higher base than other IoT
        
        return {
            'cpu-cycles': int((2200000 + (base * 37) % 900000) * multiplier),
            'instructions': int((1900000 + (base * 29) % 800000) * multiplier),
            'branch-instructions': int((140000 + (base * 17) % 70000) * multiplier),
            'branch-misses': int((4500 + (base * 13) % 3000) * multiplier),
            'cache-references': int((200000 + (base * 23) % 100000) * multiplier),
            'cache-misses': int((11000 + (base * 19) % 8000) * multiplier),
            'L1-dcache-loads': int((170000 + (base * 31) % 90000) * multiplier),
            'L1-dcache-load-misses': int((8000 + (base * 11) % 6000) * multiplier),
            'LLC-loads': int((38000 + (base * 7) % 28000) * multiplier),
            'LLC-load-misses': int((2200 + (base * 5) % 1400) * multiplier),
            'stalled-cycles-frontend': int((190000 + (base * 41) % 130000) * multiplier),
            'stalled-cycles-backend': int((160000 + (base * 43) % 110000) * multiplier),
            'bus-cycles': int((70000 + (base * 15) % 35000) * multiplier)
        }

    def calculate_hardware_metrics(self, raw_counters):
        """Calculate hardware performance metrics for Omada Controller"""
        cycles = max(1, raw_counters.get('cpu-cycles', 1))
        instructions = raw_counters.get('instructions', 1800000)
        branches = max(1, raw_counters.get('branch-instructions', 120000))
        branch_misses = raw_counters.get('branch-misses', 3800)
        cache_refs = max(1, raw_counters.get('cache-references', 180000))
        cache_misses = raw_counters.get('cache-misses', 9500)
        l1_loads = max(1, raw_counters.get('L1-dcache-loads', 150000))
        l1_misses = raw_counters.get('L1-dcache-load-misses', 7000)
        llc_loads = max(1, raw_counters.get('LLC-loads', 35000))
        llc_misses = raw_counters.get('LLC-load-misses', 1900)
        frontend_stalls = raw_counters.get('stalled-cycles-frontend', 160000)
        backend_stalls = raw_counters.get('stalled-cycles-backend', 140000)
        
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
        """Get system context with network parameters"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = float(f.read().strip()) / 1000.0
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
                cpu_freq = float(f.read().strip()) / 1000.0
            return {
                'cpu_temp': cpu_temp, 
                'cpu_freq': cpu_freq,
                'cpu_utilization': self.cpu_utilization,
                'memory_usage_mb': self.memory_usage
            }
        except:
            return {
                'cpu_temp': 42.0, 
                'cpu_freq': 1200.0,
                'cpu_utilization': self.cpu_utilization,
                'memory_usage_mb': self.memory_usage
            }

    def collect_sample(self):
        """Collect one comprehensive Omada Controller sample"""
        timestamp = datetime.now()
        
        # Simulate Omada Controller workload
        workload_raw, workload_intensity = self.simulate_omada_workload()
        
        # Read hardware counters
        hw_counters = self.read_hardware_performance_counters()
        
        # Calculate hardware metrics
        hw_metrics = self.calculate_hardware_metrics(hw_counters)
        
        # Get system context
        system_info = self.get_system_context()
        
        # Omada Controller specific metrics
        omada_metrics = {
            'connected_devices': self.connected_devices,
            'managed_aps': self.managed_aps,
            'network_throughput_mbps': self.network_throughput,
            'connected_clients': self.connected_clients,
            'security_events': self.security_events,
            'packet_loss_rate': self.packet_loss_rate,
            'wifi_interference': self.wifi_interference
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
        sample.update(omada_metrics)
        
        return sample

    def run_collection(self):
        """Main collection loop for Omada Controller"""
        print("=== TP-LINK OMADA CONTROLLER HARDWARE PERFORMANCE COUNTERS ===")
        print("Simulating TP-Link Omada Controller on Raspberry Pi")
        print(f"Target: {self.samples} samples in {self.duration} seconds")
        print(f"Hardware counters available: {len(self.working_events)}")
        print("-" * 60)
        
        start_time = time.time()
        security_count = 0
        traffic_analysis_count = 0
        
        for i in range(self.samples):
            sample_start = time.time()
            
            sample = self.collect_sample()
            self.data.append(sample)
            
            # Count network events
            if sample['workload_state'] == 5:  # Security monitoring
                security_count += 1
            if sample['workload_state'] == 11:  # Traffic analysis
                traffic_analysis_count += 1
            
            if i % 500 == 0:
                elapsed = time.time() - start_time
                progress = (i / self.samples) * 100
                security_percent = (security_count / (i + 1)) * 100
                traffic_percent = (traffic_analysis_count / (i + 1)) * 100
                
                print(f"Sample {i:5d}/{self.samples} ({progress:5.1f}%)")
                print(f"  State: {self.current_state:28} | "
                      f"Intensity: {sample['workload_intensity']:5.1f}")
                print(f"  Security: {security_percent:5.1f}% | "
                      f"Traffic Analysis: {traffic_percent:5.1f}% | "
                      f"Clients: {sample['connected_clients']:3d}")
                print(f"  Throughput: {sample['network_throughput_mbps']:4.0f} Mbps | "
                      f"APs: {sample['managed_aps']:2d} | "
                      f"Packet Loss: {sample['packet_loss_rate']:5.2f}%")
                print(f"  IPC: {sample['hw_instructions_per_cycle']:5.3f} | "
                      f"CPU Util: {sample['cpu_utilization']:5.1f}% | "
                      f"Memory: {sample['memory_usage_mb']:4.0f} MB")
                print()
            
            elapsed_sample = time.time() - sample_start
            time.sleep(max(0.001, self.interval - elapsed_sample))
        
        self.save_data()

    def save_data(self):
        """Save collected Omada Controller data to CSV"""
        filename = f"tplink_omada_hpc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
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
        security_samples = sum(1 for s in self.data if s['workload_state'] == 5)
        traffic_samples = sum(1 for s in self.data if s['workload_state'] == 11)
        security_percent = (security_samples / total_samples) * 100
        traffic_percent = (traffic_samples / total_samples) * 100
        avg_intensity = sum(s['workload_intensity'] for s in self.data) / total_samples
        avg_ipc = sum(s['hw_instructions_per_cycle'] for s in self.data) / total_samples
        
        print(f"Saved {total_samples} samples to {filename}")
        print(f"Security monitoring samples: {security_samples}/{total_samples} ({security_percent:.1f}%)")
        print(f"Traffic analysis samples: {traffic_samples}/{total_samples} ({traffic_percent:.1f}%)")
        print(f"Average workload intensity: {avg_intensity:.1f}")
        print(f"Average Instructions Per Cycle: {avg_ipc:.3f}")
        
        # Network specific statistics
        avg_throughput = sum(s['network_throughput_mbps'] for s in self.data) / total_samples
        avg_clients = sum(s['connected_clients'] for s in self.data) / total_samples
        print(f"Average Network Throughput: {avg_throughput:.1f} Mbps")
        print(f"Average Connected Clients: {avg_clients:.1f}")

if __name__ == "__main__":
    # Enable performance monitoring
    os.system("echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid > /dev/null 2>&1")
    collector = TPLinkOmadaHPC(samples=20000, duration=600)
    collector.run_collection()
