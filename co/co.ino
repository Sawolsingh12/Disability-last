/*
  Assistive Mobility System - Arduino/ESP32 Code
  Compatible with the Python Data Analysis & Machine Learning Module
  
  Hardware Requirements:
  - ESP32 or Arduino with IMU sensor (MPU6050)
  - Ultrasonic sensor (HC-SR04) for distance measurement
  - Motor driver (L298N) for wheelchair control
  - Push buttons for mode switching
  - LED indicators for status
  
  Author: Assistive Mobility System
  Version: 1.0
*/

#include <Wire.h>
#include <MPU6050.h>

// Pin Definitions
#define TRIG_PIN 5          // Ultrasonic sensor trigger pin
#define ECHO_PIN 18         // Ultrasonic sensor echo pin
#define MOTOR_LEFT_A 25     // Left motor pin A
#define MOTOR_LEFT_B 26     // Left motor pin B
#define MOTOR_RIGHT_A 27    // Right motor pin A
#define MOTOR_RIGHT_B 14    // Right motor pin B
#define MOTOR_LEFT_PWM 12   // Left motor PWM (speed control)
#define MOTOR_RIGHT_PWM 13  // Right motor PWM (speed control)

// Control buttons
#define MODE_BUTTON 4       // Mode switching button
#define EMERGENCY_BUTTON 2  // Emergency stop button

// LED indicators
#define STATUS_LED 23       // General status LED
#define OBSTACLE_LED 22     // Obstacle warning LED
#define MODE_LED_1 21       // Mode indicator LED 1
#define MODE_LED_2 19       // Mode indicator LED 2

// System Constants
#define OBSTACLE_THRESHOLD 30.0  // Obstacle detection threshold in cm
#define MAX_SPEED 255               // Maximum motor speed
#define MIN_SPEED 80                // Minimum motor speed for movement
#define TILT_SENSITIVITY 15.0       // Tilt control sensitivity
#define SERIAL_BAUD 115200         // Serial communication baud rate
#define DATA_INTERVAL 100          // Data transmission interval (ms)

// Control Modes
enum ControlMode {
  MANUAL = 0,      // Manual joystick/button control
  TILT = 1,        // Head tilt control
  OBSTACLE_AWARE = 2  // Automatic obstacle avoidance
};

// Global Variables
MPU6050 mpu;
ControlMode currentMode = MANUAL;
bool emergencyStop = false;
bool obstacleDetected = false;
float currentSpeed = 0;
float tiltX = 0, tiltY = 0;
float distance = 100.0;
unsigned long lastDataTime = 0;
unsigned long lastButtonCheck = 0;
bool lastModeButtonState = HIGH;
bool lastEmergencyButtonState = HIGH;

// Motor control variables
int leftMotorSpeed = 0;
int rightMotorSpeed = 0;

void setup() {
  Serial.begin(SERIAL_BAUD);
  Wire.begin();
  
  // Initialize MPU6050
  mpu.initialize();
  if (mpu.testConnection()) {
    Serial.println("MPU6050 connection successful");
  } else {
    Serial.println("MPU6050 connection failed");
  }
  
  // Configure pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(MODE_BUTTON, INPUT_PULLUP);
  pinMode(EMERGENCY_BUTTON, INPUT_PULLUP);
  
  // Motor pins
  pinMode(MOTOR_LEFT_A, OUTPUT);
  pinMode(MOTOR_LEFT_B, OUTPUT);
  pinMode(MOTOR_RIGHT_A, OUTPUT);
  pinMode(MOTOR_RIGHT_B, OUTPUT);
  pinMode(MOTOR_LEFT_PWM, OUTPUT);
  pinMode(MOTOR_RIGHT_PWM, OUTPUT);
  
  // LED pins
  pinMode(STATUS_LED, OUTPUT);
  pinMode(OBSTACLE_LED, OUTPUT);
  pinMode(MODE_LED_1, OUTPUT);
  pinMode(MODE_LED_2, OUTPUT);
  
  // Initialize system
  stopMotors();
  updateModeIndicators();
  digitalWrite(STATUS_LED, HIGH);
  
  Serial.println("🧑‍🦽 Assistive Mobility System Initialized");
  Serial.println("Modes: 0=Manual, 1=Tilt, 2=Obstacle-Aware");
  delay(1000);
}

void loop() {
  // Check for emergency stop
  checkEmergencyButton();
  
  if (!emergencyStop) {
    // Read sensors
    readIMUSensor();
    distance = readUltrasonicSensor();
    obstacleDetected = (distance < OBSTACLE_THRESHOLD);
    
    // Check mode button
    checkModeButton();
    
    // Execute control logic based on current mode
    switch (currentMode) {
      case MANUAL:
        executeManualControl();
        break;
      case TILT:
        executeTiltControl();
        break;
      case OBSTACLE_AWARE:
        executeObstacleAwareControl();
        break;
    }
    
    // Update motor speeds
    updateMotors();
  } else {
    // Emergency stop - all motors off
    stopMotors();
    currentSpeed = 0;
  }
  
  // Update status indicators
  updateStatusLEDs();
  
  // Send data to Python system
  if (millis() - lastDataTime >= DATA_INTERVAL) {
    sendSensorData();
    lastDataTime = millis();
  }
  
  // Process any incoming commands from Python
  processSerialCommands();
  
  delay(10); // Small delay for system stability
}

void readIMUSensor() {
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  
  // Convert to degrees (simplified calculation)
  tiltX = atan2(ay, az) * 180.0 / PI;
  tiltY = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / PI;
  
  // Apply smoothing filter
  static float lastTiltX = 0, lastTiltY = 0;
  tiltX = 0.8 * lastTiltX + 0.2 * tiltX;
  tiltY = 0.8 * lastTiltY + 0.2 * tiltY;
  lastTiltX = tiltX;
  lastTiltY = tiltY;
}

float readUltrasonicSensor() {
  // Send trigger pulse
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  // Read echo pulse
  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // 30ms timeout
  
  if (duration == 0) {
    return 200.0; // No obstacle detected (max range)
  }
  
  // Calculate distance in cm
  float dist = (duration * 0.034) / 2.0;
  
  // Apply smoothing filter
  static float lastDistance = 100.0;
  dist = 0.7 * lastDistance + 0.3 * dist;
  lastDistance = dist;
  
  return constrain(dist, 5.0, 200.0);
}

void checkModeButton() {
  if (millis() - lastButtonCheck > 50) { // Debounce delay
    bool currentButtonState = digitalRead(MODE_BUTTON);
    
    if (currentButtonState == LOW && lastModeButtonState == HIGH) {
      // Button pressed
      currentMode = (ControlMode)((currentMode + 1) % 3);
      updateModeIndicators();
      Serial.print("Mode changed to: ");
      Serial.println(currentMode);
      delay(200); // Additional debounce
    }
    
    lastModeButtonState = currentButtonState;
    lastButtonCheck = millis();
  }
}

void checkEmergencyButton() {
  bool currentEmergencyState = digitalRead(EMERGENCY_BUTTON);
  
  if (currentEmergencyState == LOW && lastEmergencyButtonState == HIGH) {
    emergencyStop = !emergencyStop;
    Serial.print("Emergency stop: ");
    Serial.println(emergencyStop ? "ACTIVATED" : "DEACTIVATED");
    delay(200);
  }
  
  lastEmergencyButtonState = currentEmergencyState;
}

void executeManualControl() {
  // In manual mode, use basic forward movement
  // In a real implementation, this would read joystick inputs
  currentSpeed = 150; // Default moderate speed
  leftMotorSpeed = currentSpeed;
  rightMotorSpeed = currentSpeed;
}

void executeTiltControl() {
  // Use head tilt to control movement
  float forwardTilt = constrain(tiltY, -30, 30);
  float sideTilt = constrain(tiltX, -30, 30);
  
  // Convert tilt to speed (forward/backward)
  if (abs(forwardTilt) > 5) { // Dead zone
    currentSpeed = map(abs(forwardTilt), 5, 30, MIN_SPEED, MAX_SPEED);
    if (forwardTilt < 0) currentSpeed = -currentSpeed; // Reverse
  } else {
    currentSpeed = 0;
  }
  
  // Convert side tilt to differential steering
  float steerFactor = sideTilt / 30.0; // -1 to 1
  
  if (currentSpeed != 0) {
    leftMotorSpeed = currentSpeed * (1.0 - steerFactor * 0.5);
    rightMotorSpeed = currentSpeed * (1.0 + steerFactor * 0.5);
  } else {
    // In-place turning
    if (abs(sideTilt) > 10) {
      leftMotorSpeed = steerFactor * MIN_SPEED;
      rightMotorSpeed = -steerFactor * MIN_SPEED;
    } else {
      leftMotorSpeed = 0;
      rightMotorSpeed = 0;
    }
  }
}

void executeObstacleAwareControl() {
  // Start with tilt control
  executeTiltControl();
  
  // Apply obstacle avoidance
  if (obstacleDetected) {
    if (currentSpeed > 0) { // Only stop forward movement
      currentSpeed = 0;
      leftMotorSpeed = 0;
      rightMotorSpeed = 0;
    }
    
    // If very close to obstacle, back up slightly
    if (distance < 15) {
      currentSpeed = -MIN_SPEED;
      leftMotorSpeed = currentSpeed;
      rightMotorSpeed = currentSpeed;
    }
  }
  
  // Gradual speed reduction as approaching obstacles
  if (distance < 50 && currentSpeed > 0) {
    float speedReduction = map(distance, 10, 50, 0.3, 1.0);
    leftMotorSpeed *= speedReduction;
    rightMotorSpeed *= speedReduction;
    currentSpeed *= speedReduction;
  }
}

void updateMotors() {
  // Constrain motor speeds
  leftMotorSpeed = constrain(leftMotorSpeed, -MAX_SPEED, MAX_SPEED);
  rightMotorSpeed = constrain(rightMotorSpeed, -MAX_SPEED, MAX_SPEED);
  
  // Control left motor
  if (leftMotorSpeed > 0) {
    digitalWrite(MOTOR_LEFT_A, HIGH);
    digitalWrite(MOTOR_LEFT_B, LOW);
  } else if (leftMotorSpeed < 0) {
    digitalWrite(MOTOR_LEFT_A, LOW);
    digitalWrite(MOTOR_LEFT_B, HIGH);
  } else {
    digitalWrite(MOTOR_LEFT_A, LOW);
    digitalWrite(MOTOR_LEFT_B, LOW);
  }
  analogWrite(MOTOR_LEFT_PWM, abs(leftMotorSpeed));
  
  // Control right motor
  if (rightMotorSpeed > 0) {
    digitalWrite(MOTOR_RIGHT_A, HIGH);
    digitalWrite(MOTOR_RIGHT_B, LOW);
  } else if (rightMotorSpeed < 0) {
    digitalWrite(MOTOR_RIGHT_A, LOW);
    digitalWrite(MOTOR_RIGHT_B, HIGH);
  } else {
    digitalWrite(MOTOR_RIGHT_A, LOW);
    digitalWrite(MOTOR_RIGHT_B, LOW);
  }
  analogWrite(MOTOR_RIGHT_PWM, abs(rightMotorSpeed));
}

void stopMotors() {
  digitalWrite(MOTOR_LEFT_A, LOW);
  digitalWrite(MOTOR_LEFT_B, LOW);
  digitalWrite(MOTOR_RIGHT_A, LOW);
  digitalWrite(MOTOR_RIGHT_B, LOW);
  analogWrite(MOTOR_LEFT_PWM, 0);
  analogWrite(MOTOR_RIGHT_PWM, 0);
  leftMotorSpeed = 0;
  rightMotorSpeed = 0;
}

void updateModeIndicators() {
  // Update mode LEDs
  digitalWrite(MODE_LED_1, currentMode & 1);
  digitalWrite(MODE_LED_2, (currentMode >> 1) & 1);
}

void updateStatusLEDs() {
  // Obstacle warning LED
  digitalWrite(OBSTACLE_LED, obstacleDetected);
  
  // Status LED blinks if emergency stop is active
  if (emergencyStop) {
    digitalWrite(STATUS_LED, (millis() / 200) % 2);
  } else {
    digitalWrite(STATUS_LED, HIGH);
  }
}

void sendSensorData() {
  // Send data in format expected by Python system
  Serial.print("Mode:");
  Serial.print(currentMode);
  Serial.print(",Speed:");
  Serial.print(abs(currentSpeed));
  Serial.print(",TiltX:");
  Serial.print(tiltX, 1);
  Serial.print(",TiltY:");
  Serial.print(tiltY, 1);
  Serial.print(",Distance:");
  Serial.print(distance, 1);
  Serial.print(",Obstacle:");
  Serial.println(obstacleDetected ? "YES" : "NO");
}

void processSerialCommands() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    // Process commands from Python system
    if (command.startsWith("MODE:")) {
      int newMode = command.substring(5).toInt();
      if (newMode >= 0 && newMode <= 2) {
        currentMode = (ControlMode)newMode;
        updateModeIndicators();
        Serial.print("Mode set to: ");
        Serial.println(currentMode);
      }
    }
    else if (command == "EMERGENCY_STOP") {
      emergencyStop = true;
      Serial.println("Emergency stop activated via serial");
    }
    else if (command == "EMERGENCY_RESET") {
      emergencyStop = false;
      Serial.println("Emergency stop deactivated via serial");
    }
    else if (command == "STATUS") {
      Serial.print("Status - Mode: ");
      Serial.print(currentMode);
      Serial.print(", Emergency: ");
      Serial.print(emergencyStop ? "ON" : "OFF");
      Serial.print(", Obstacle: ");
      Serial.println(obstacleDetected ? "YES" : "NO");
    }
    else if (command.startsWith("SPEED:")) {
      // Manual speed override (for testing)
      int newSpeed = command.substring(6).toInt();
      currentSpeed = constrain(newSpeed, -MAX_SPEED, MAX_SPEED);
      Serial.print("Speed set to: ");
      Serial.println(currentSpeed);
    }
  }
}

// Additional helper functions for advanced features

void calibrateIMU() {
  Serial.println("Calibrating IMU... Keep device still");
  delay(2000);
  
  // Simple calibration - average readings for offset
  float offsetX = 0, offsetY = 0;
  for (int i = 0; i < 100; i++) {
    readIMUSensor();
    offsetX += tiltX;
    offsetY += tiltY;
    delay(10);
  }
  
  // In a real implementation, store these offsets in EEPROM
  Serial.println("IMU calibration complete");
}

void performSystemTest() {
  Serial.println("🔧 System Test Mode");
  
  // Test LEDs
  Serial.println("Testing LEDs...");
  digitalWrite(STATUS_LED, HIGH);
  delay(500);
  digitalWrite(OBSTACLE_LED, HIGH);
  delay(500);
  digitalWrite(MODE_LED_1, HIGH);
  delay(500);
  digitalWrite(MODE_LED_2, HIGH);
  delay(500);
  
  // Turn off all LEDs
  digitalWrite(STATUS_LED, LOW);
  digitalWrite(OBSTACLE_LED, LOW);
  digitalWrite(MODE_LED_1, LOW);
  digitalWrite(MODE_LED_2, LOW);
  delay(500);
  
  // Test motors (brief movement)
  Serial.println("Testing motors...");
  leftMotorSpeed = 100;
  rightMotorSpeed = 100;
  updateMotors();
  delay(1000);
  stopMotors();
  
  // Test sensors
  Serial.println("Testing sensors...");
  for (int i = 0; i < 10; i++) {
    readIMUSensor();
    float dist = readUltrasonicSensor();
    Serial.print("IMU: X=");
    Serial.print(tiltX);
    Serial.print(", Y=");
    Serial.print(tiltY);
    Serial.print(" | Distance: ");
    Serial.println(dist);
    delay(200);
  }
  
  Serial.println("✅ System test complete");
  updateModeIndicators();
}