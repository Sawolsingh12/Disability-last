#!/usr/bin/env python3
"""
Assistive Mobility System - Data Analysis & Machine Learning Module
Processes sensor data, detects patterns, and provides predictive analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import serial
import json
import time
import threading
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MobilityDataAnalyzer:
    def __init__(self, serial_port=None, baud_rate=115200):
        """Initialize the data analyzer with serial connection to ESP32"""
        # Auto-detect Windows COM ports or use provided port
        if serial_port is None:
            import platform
            if platform.system() == 'Windows':
                # Try common Windows COM ports
                for i in range(1, 20):
                    try:
                        test_port = f'COM{i}'
                        test_connection = serial.Serial(test_port, baud_rate, timeout=0.1)
                        test_connection.close()
                        self.serial_port = test_port
                        print(f"🔍 Auto-detected serial port: {test_port}")
                        break
                    except:
                        continue
                else:
                    self.serial_port = 'COM3'  # Default fallback
            else:
                self.serial_port = '/dev/ttyUSB0'  # Linux/Mac default
        else:
            self.serial_port = serial_port
            
        self.baud_rate = baud_rate
        self.connection = None
        self.data_buffer = []
        self.is_collecting = False
        
        # Machine learning models
        self.obstacle_predictor = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        
        # Data storage
        self.sensor_data = pd.DataFrame()
        self.user_patterns = {}
        
        # Setup data collection
        self.setup_serial_connection()
        
    def setup_serial_connection(self):
        """Establish serial connection with ESP32"""
        try:
            self.connection = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            print(f"✅ Connected to ESP32 on {self.serial_port}")
            time.sleep(2)  # Allow connection to stabilize
        except serial.SerialException as e:
            print(f"❌ Failed to connect to ESP32: {e}")
            print("📝 Switching to simulation mode for demo purposes")
            self.connection = None
    
    def start_data_collection(self):
        """Start collecting data from the mobility system"""
        self.is_collecting = True
        collection_thread = threading.Thread(target=self._collect_data_loop)
        collection_thread.daemon = True
        collection_thread.start()
        print("📊 Data collection started")
    
    def stop_data_collection(self):
        """Stop data collection"""
        self.is_collecting = False
        if self.connection:
            self.connection.close()
        print("🛑 Data collection stopped")
    
    def _collect_data_loop(self):
        """Main data collection loop"""
        while self.is_collecting:
            try:
                if self.connection and self.connection.in_waiting:
                    # Read data from ESP32
                    line = self.connection.readline().decode('utf-8').strip()
                    if line.startswith('Mode:'):
                        data_point = self._parse_sensor_data(line)
                        if data_point:
                            self.data_buffer.append(data_point)
                else:
                    # Simulate data for demo purposes
                    data_point = self._generate_simulated_data()
                    self.data_buffer.append(data_point)
                
                # Process buffer every 10 data points
                if len(self.data_buffer) >= 10:
                    self._process_data_buffer()
                
                time.sleep(0.1)  # 10Hz data collection
                
            except Exception as e:
                print(f"⚠️ Data collection error: {e}")
                time.sleep(1)
    
    def _parse_sensor_data(self, line):
        """Parse sensor data from ESP32 serial output"""
        try:
            # Parse: "Mode:1,Speed:150,TiltX:2.3,TiltY:-1.2,Distance:45,Obstacle:NO"
            parts = line.split(',')
            data = {}
            
            for part in parts:
                key, value = part.split(':')
                if key == 'Obstacle':
                    data[key] = 1 if value == 'YES' else 0
                else:
                    data[key] = float(value)
            
            data['timestamp'] = datetime.now()
            return data
            
        except Exception as e:
            print(f"⚠️ Data parsing error: {e}")
            return None
    
    def _generate_simulated_data(self):
        """Generate simulated sensor data for demo purposes"""
        current_time = datetime.now()
        
        # Simulate realistic mobility patterns
        base_speed = 150 + np.random.normal(0, 20)
        tilt_x = np.random.normal(0, 5)  # Head tilt left/right
        tilt_y = np.random.normal(2, 3)  # Head tilt forward/back
        distance = np.random.exponential(100) + 20  # Distance to obstacles
        
        # Simulate occasional obstacles
        obstacle = 1 if distance < 30 else 0
        
        # Add some correlated patterns (realistic user behavior)
        if abs(tilt_x) > 10:  # Sharp turn
            base_speed *= 0.7  # Slow down during turns
        
        return {
            'timestamp': current_time,
            'Mode': np.random.choice([0, 1, 2]),  # Manual, Tilt, Auto
            'Speed': max(0, min(255, base_speed)),
            'TiltX': tilt_x,
            'TiltY': tilt_y,
            'Distance': distance,
            'Obstacle': obstacle
        }
    
    def _process_data_buffer(self):
        """Process accumulated data buffer"""
        df_new = pd.DataFrame(self.data_buffer)
        self.sensor_data = pd.concat([self.sensor_data, df_new], ignore_index=True)
        self.data_buffer.clear()
        
        # Keep only last 1000 data points for memory efficiency
        if len(self.sensor_data) > 1000:
            self.sensor_data = self.sensor_data.tail(1000).reset_index(drop=True)
        
        # Trigger analysis if we have enough data
        if len(self.sensor_data) >= 100:
            self._update_models()
    
    def _update_models(self):
        """Update machine learning models with new data"""
        try:
            # Prepare features for ML
            features = ['Speed', 'TiltX', 'TiltY', 'Distance']
            X = self.sensor_data[features].fillna(0)
            
            # Update obstacle prediction model
            if 'Obstacle' in self.sensor_data.columns:
                y_obstacle = self.sensor_data['Obstacle']
                self._train_obstacle_predictor(X, y_obstacle)
            
            # Update anomaly detection model
            self._train_anomaly_detector(X)
            
        except Exception as e:
            print(f"⚠️ Model update error: {e}")
    
    def _train_obstacle_predictor(self, X, y):
        """Train obstacle prediction model"""
        if len(X) < 50:
            return
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        self.obstacle_predictor = RandomForestClassifier(
            n_estimators=50, random_state=42, max_depth=10
        )
        self.obstacle_predictor.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.obstacle_predictor.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        print(f"🎯 Obstacle prediction accuracy: {accuracy:.3f}")
    
    def _train_anomaly_detector(self, X):
        """Train anomaly detection model"""
        if len(X) < 50:
            return
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest for anomaly detection
        self.anomaly_detector = IsolationForest(
            contamination=0.1, random_state=42
        )
        self.anomaly_detector.fit(X_scaled)
        print("🔍 Anomaly detection model updated")
    
    def predict_obstacle_risk(self, speed, tilt_x, tilt_y, distance):
        """Predict obstacle collision risk"""
        if not self.obstacle_predictor:
            return 0.5  # Unknown risk
        
        features = np.array([[speed, tilt_x, tilt_y, distance]])
        features_scaled = self.scaler.transform(features)
        
        risk_prob = self.obstacle_predictor.predict_proba(features_scaled)[0][1]
        return risk_prob
    
    def detect_anomaly(self, speed, tilt_x, tilt_y, distance):
        """Detect if current sensor readings are anomalous"""
        if not self.anomaly_detector:
            return False, 0.0  # Return tuple with default values
        
        try:
            features = np.array([[speed, tilt_x, tilt_y, distance]])
            features_scaled = self.scaler.transform(features)
            
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
            
            return is_anomaly, anomaly_score
        except Exception as e:
            print(f"⚠️ Anomaly detection error: {e}")
            return False, 0.0
    
    def analyze_usage_patterns(self):
        """Analyze user behavior patterns"""
        if len(self.sensor_data) < 50:
            return {}
        
        patterns = {}
        
        # Mode preferences
        mode_counts = self.sensor_data['Mode'].value_counts()
        patterns['preferred_mode'] = mode_counts.index[0]
        patterns['mode_distribution'] = mode_counts.to_dict()
        
        # Speed patterns
        patterns['avg_speed'] = self.sensor_data['Speed'].mean()
        patterns['speed_variance'] = self.sensor_data['Speed'].var()
        
        # Tilt sensitivity
        patterns['tilt_sensitivity_x'] = self.sensor_data['TiltX'].std()
        patterns['tilt_sensitivity_y'] = self.sensor_data['TiltY'].std()
        
        # Obstacle encounter frequency
        obstacle_rate = self.sensor_data['Obstacle'].mean()
        patterns['obstacle_encounter_rate'] = obstacle_rate
        
        # Time-based patterns
        self.sensor_data['hour'] = pd.to_datetime(self.sensor_data['timestamp']).dt.hour
        patterns['active_hours'] = self.sensor_data['hour'].value_counts().to_dict()
        
        return patterns
    
    def generate_recommendations(self):
        """Generate personalized recommendations based on usage patterns"""
        patterns = self.analyze_usage_patterns()
        recommendations = []
        
        if not patterns:
            return ["Collect more data to generate personalized recommendations"]
        
        # Speed recommendations
        if patterns.get('avg_speed', 0) > 200:
            recommendations.append("Consider reducing average speed for safer navigation")
        elif patterns.get('avg_speed', 0) < 100:
            recommendations.append("You might benefit from slightly higher speeds for efficiency")
        
        # Mode recommendations
        obstacle_rate = patterns.get('obstacle_encounter_rate', 0)
        if obstacle_rate > 0.3:
            recommendations.append("High obstacle encounter rate - consider using Auto mode more often")
        
        # Tilt sensitivity
        tilt_sens_x = patterns.get('tilt_sensitivity_x', 0)
        if tilt_sens_x > 15:
            recommendations.append("High tilt variance detected - consider calibrating head controls")
        
        # Usage time recommendations
        active_hours = patterns.get('active_hours', {})
        if active_hours:
            peak_hour = max(active_hours.keys(), key=lambda k: active_hours[k])
            recommendations.append(f"Peak usage at {peak_hour}:00 - ensure battery is fully charged")
        
        return recommendations
    
    def create_safety_report(self):
        """Generate comprehensive safety report"""
        if len(self.sensor_data) < 10:
            return "Insufficient data for safety report"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_points_analyzed': len(self.sensor_data),
            'safety_metrics': {}
        }
        
        # Obstacle encounters
        obstacle_data = self.sensor_data[self.sensor_data['Obstacle'] == 1]
        report['safety_metrics']['obstacle_encounters'] = len(obstacle_data)
        
        # Emergency stops (speed dropping to 0 quickly)
        speed_changes = self.sensor_data['Speed'].diff()
        emergency_stops = (speed_changes < -50).sum()
        report['safety_metrics']['emergency_stops'] = emergency_stops
        
        # Stability metrics
        tilt_x_stability = self.sensor_data['TiltX'].std()
        tilt_y_stability = self.sensor_data['TiltY'].std()
        report['safety_metrics']['stability_score'] = max(0, 100 - (tilt_x_stability + tilt_y_stability))
        
        # Risk assessment
        if self.obstacle_predictor:
            recent_data = self.sensor_data.tail(10)
            risk_scores = []
            for _, row in recent_data.iterrows():
                risk = self.predict_obstacle_risk(
                    row['Speed'], row['TiltX'], row['TiltY'], row['Distance']
                )
                risk_scores.append(risk)
            report['safety_metrics']['current_risk_level'] = np.mean(risk_scores)
        
        return report
    
    def visualize_data(self, save_plots=True):
        """Create visualizations of the collected data"""
        if len(self.sensor_data) < 10:
            print("Insufficient data for visualization")
            return
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Assistive Mobility System - Data Analysis Dashboard', fontsize=16)
        
        # 1. Speed over time
        axes[0, 0].plot(self.sensor_data.index, self.sensor_data['Speed'], alpha=0.7)
        axes[0, 0].set_title('Speed Over Time')
        axes[0, 0].set_ylabel('Speed (PWM)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Tilt patterns
        axes[0, 1].scatter(self.sensor_data['TiltX'], self.sensor_data['TiltY'], 
                          c=self.sensor_data['Speed'], cmap='viridis', alpha=0.6)
        axes[0, 1].set_title('Head Tilt Patterns')
        axes[0, 1].set_xlabel('Tilt X (degrees)')
        axes[0, 1].set_ylabel('Tilt Y (degrees)')
        
        # 3. Distance distribution
        axes[0, 2].hist(self.sensor_data['Distance'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 2].axvline(x=30, color='red', linestyle='--', label='Obstacle threshold')
        axes[0, 2].set_title('Distance to Obstacles Distribution')
        axes[0, 2].set_xlabel('Distance (cm)')
        axes[0, 2].legend()
        
        # 4. Mode usage
        mode_names = ['Manual', 'Tilt', 'Obstacle-Aware']
        mode_counts = self.sensor_data['Mode'].value_counts().sort_index()
        axes[1, 0].pie(mode_counts.values, labels=[mode_names[i] for i in mode_counts.index], 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Control Mode Usage')
        
        # 5. Safety events timeline
        obstacle_events = self.sensor_data[self.sensor_data['Obstacle'] == 1]
        axes[1, 1].scatter(obstacle_events.index, obstacle_events['Distance'], 
                          color='red', s=50, alpha=0.7)
        axes[1, 1].set_title('Obstacle Detection Events')
        axes[1, 1].set_xlabel('Time Index')
        axes[1, 1].set_ylabel('Distance at Detection (cm)')
        
        # 6. Feature correlation heatmap
        features = ['Speed', 'TiltX', 'TiltY', 'Distance']
        correlation_matrix = self.sensor_data[features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                   center=0, ax=axes[1, 2])
        axes[1, 2].set_title('Feature Correlations')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'mobility_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved as mobility_analysis_{timestamp}.png")
        
        plt.show()
    
    def export_data(self, filename=None):
        """Export collected data to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'mobility_data_{timestamp}.csv'
        
        self.sensor_data.to_csv(filename, index=False)
        print(f"💾 Data exported to {filename}")
        return filename
    
    def real_time_monitor(self, duration_minutes=5):
        """Run real-time monitoring and analysis"""
        print(f"🔍 Starting {duration_minutes}-minute real-time monitoring session")
        print("📊 Collecting initial data...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        # Allow some time to collect initial data
        time.sleep(2)
        
        while time.time() < end_time:
            if len(self.sensor_data) > 0:
                latest_data = self.sensor_data.iloc[-1]
                
                # Check for anomalies (only if we have enough data for the model)
                if len(self.sensor_data) >= 50:
                    is_anomaly, anomaly_score = self.detect_anomaly(
                        latest_data['Speed'], latest_data['TiltX'], 
                        latest_data['TiltY'], latest_data['Distance']
                    )
                    
                    # Predict obstacle risk
                    obstacle_risk = self.predict_obstacle_risk(
                        latest_data['Speed'], latest_data['TiltX'],
                        latest_data['TiltY'], latest_data['Distance']
                    )
                else:
                    is_anomaly, anomaly_score = False, 0.0
                    obstacle_risk = 0.5  # Unknown risk
                
                # Display real-time status
                status = f"\r⏱️  Speed: {latest_data['Speed']:.0f} | "
                status += f"Distance: {latest_data['Distance']:.0f}cm | "
                status += f"Risk: {obstacle_risk:.2f} | "
                status += f"Anomaly: {'YES' if is_anomaly else 'NO'} | "
                status += f"Points: {len(self.sensor_data)}"
                
                print(status, end="", flush=True)
                
                # Alert for high risk situations
                if obstacle_risk > 0.8:
                    print(f"\n🚨 HIGH COLLISION RISK DETECTED! Risk: {obstacle_risk:.3f}")
                
                if is_anomaly and anomaly_score < -0.5:
                    print(f"\n⚠️  ANOMALY DETECTED! Score: {anomaly_score:.3f}")
            else:
                print(f"\r📊 Collecting data... (Points: {len(self.sensor_data)})", end="", flush=True)
            
            time.sleep(1)
        
        print(f"\n✅ Monitoring session completed - Collected {len(self.sensor_data)} data points")

def main():
    """Main function to demonstrate the data analysis system"""
    print("🧑‍🦽 Assistive Mobility System - Data Analysis Module")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = MobilityDataAnalyzer()
    
    try:
        # Start data collection
        analyzer.start_data_collection()
        
        # Run real-time monitoring for 2 minutes (demo)
        analyzer.real_time_monitor(duration_minutes=2)
        
        # Generate analysis
        print("\n📊 Generating usage analysis...")
        patterns = analyzer.analyze_usage_patterns()
        
        if patterns:
            print("\n📈 Usage Patterns:")
            for key, value in patterns.items():
                if isinstance(value, dict):
                    print(f"  {key}: {dict(list(value.items())[:3])}")  # Show top 3
                else:
                    print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Generate recommendations
        print("\n💡 Personalized Recommendations:")
        recommendations = analyzer.generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Generate safety report
        print("\n🛡️ Safety Report:")
        safety_report = analyzer.create_safety_report()
        if isinstance(safety_report, dict):
            print(f"  Data Points Analyzed: {safety_report['data_points_analyzed']}")
            metrics = safety_report['safety_metrics']
            print(f"  Obstacle Encounters: {metrics.get('obstacle_encounters', 0)}")
            print(f"  Emergency Stops: {metrics.get('emergency_stops', 0)}")
            print(f"  Stability Score: {metrics.get('stability_score', 0):.1f}/100")
            if 'current_risk_level' in metrics:
                risk_level = metrics['current_risk_level']
                risk_status = "HIGH" if risk_level > 0.7 else "MEDIUM" if risk_level > 0.3 else "LOW"
                print(f"  Current Risk Level: {risk_level:.3f} ({risk_status})")
        
        # Create visualizations
        print("\n📊 Generating visualizations...")
        analyzer.visualize_data()
        
        # Export data
        filename = analyzer.export_data()
        print(f"📁 Raw data exported to: {filename}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    finally:
        analyzer.stop_data_collection()
        print("🏁 Analysis session completed")

if __name__ == "__main__":
    main()