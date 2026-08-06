# 3D Printer with AI-Based Real-Time Fault Detection

An open-source project showcasing the design and development of a **low-cost, high-accuracy 3D printer** built from scratch for under **₹5,000** using Arduino Mega, RAMPS 1.4, Marlin Firmware, and second-hand components. The printer is integrated with a fine-tuned **YOLOv8** computer vision model that performs **real-time print fault detection**, helping improve print reliability, reduce material waste, and demonstrate the integration of embedded systems, firmware, and artificial intelligence.

# 🖨️ Low-Cost 3D Printer with AI-Based Real-Time Fault Detection

A fully functional, high-accuracy 3D printer built completely from scratch for **under ₹5,000** using primarily second-hand components. This project combines embedded systems, electronics, firmware development, mechanical assembly, and artificial intelligence by integrating a fine-tuned **YOLOv8** model for real-time 3D print fault detection.

---

# 📖 Overview

This project was developed to demonstrate that a reliable and accurate 3D printer can be built at a very low cost without relying on expensive commercial kits. By carefully sourcing second-hand components and configuring them with open-source firmware, the printer delivers excellent print quality while maintaining a budget of less than ₹5,000.

To further improve the printing process, an AI-based monitoring system was developed using a fine-tuned YOLOv8 model capable of detecting print defects in real time. This helps reduce failed prints, save material, and improve overall print reliability.

---

# 🎯 Objectives

- Build a low-cost 3D printer from scratch.
- Achieve excellent printing accuracy.
- Minimize the overall project cost by using second-hand components.
- Configure and optimize Marlin Firmware.
- Integrate an AI-based print monitoring system.
- Detect print faults during printing using YOLOv8.
- Gain practical experience in embedded systems, hardware integration, and computer vision.

---

# ✨ Features

- Fully functional DIY 3D printer
- Total build cost under ₹5,000
- High printing accuracy
- Built primarily using second-hand components
- Arduino Mega + RAMPS 1.4 based control system
- Marlin Firmware
- BLTouch Automatic Bed Leveling
- LCD interface for printer control
- UltiMaker Cura slicing support
- Real-time AI-based print fault detection using YOLOv8
- Modular and upgrade-friendly design

---

# 🛠 Hardware Components

| Component | Purpose |
|-----------|---------|
| Arduino Mega 2560 | Main microcontroller controlling the printer |
| RAMPS 1.4 | Motor driver and printer control board |
| 12V Power Supply | Powers the complete printer |
| Stepper Motors | Control X, Y, and Z axis movement |
| Extruder Stepper Motor | Controls filament extrusion |
| Nozzle & Hotend | Melts and extrudes filament |
| Heated Bed *(Optional if used)* | Improves print adhesion |
| BLTouch Sensor | Automatic bed leveling |
| LCD Display | Monitor and control printer operations |
| End Stops | Axis homing and positioning |
| GT2 Belts & Pulleys | X and Y motion transmission |
| Lead Screw | Z-axis movement |
| Cooling Fans | Hotend and print cooling |
| Mechanical Frame | Structural support |

---

# 💻 Software Stack

## Firmware

- Marlin Firmware

Responsible for:

- Motion control
- Temperature control
- Motor control
- Endstop handling
- Auto bed leveling
- LCD interface
- G-code execution

---

## Slicer

UltiMaker Cura

Used for:

- Slicing STL models
- Generating G-code
- Print configuration
- Layer visualization

---

## Programming

- Arduino IDE
- C/C++
- Python

---

## AI Framework

- YOLOv8
- Ultralytics
- OpenCV

---

# ⚙ Working Principle

1. Design or download a 3D model.
2. Slice the model using UltiMaker Cura.
3. Generate G-code.
4. Upload the G-code to the printer.
5. Marlin Firmware interprets each command.
6. Arduino Mega controls the motors through RAMPS 1.4.
7. Stepper motors move the print head along X, Y, and Z axes.
8. The extruder melts and deposits filament layer by layer.
9. BLTouch automatically levels the print bed.
10. During printing, the YOLOv8 model continuously monitors the print and detects defects in real time.

---

# 🤖 AI-Based Print Fault Detection

To enhance printer reliability, a custom YOLOv8 model was fine-tuned to detect common 3D printing failures in real time.

## Objectives

- Detect printing defects early
- Reduce failed prints
- Save filament
- Improve print quality
- Enable intelligent monitoring

---

## Faults Detected

- Layer Cracking
- Over Extrusion
- Stringing
- Warping

---

## AI Workflow

Dataset Collection

↓

Data Annotation

↓

YOLOv8 Training

↓

Model Fine-Tuning

↓

Real-Time Camera Input

↓

Defect Detection

↓

Bounding Box Prediction

↓

Fault Classification

↓

Monitoring Output

---

# 📂 Project Structure

```
3d-printer-with-ai-fault-detection/

│── Firmware/
│ └── Marlin/

│── YOLOv8/
│ ├── Dataset/
│ ├── Training/
│ ├── Models/
│ ├── Detection/
│ └── Weights/

│── Hardware/
│ ├── Wiring/
│ ├── Components/
│ └── BOM/

│── Cura_Profile/

│── Documentation/

│── README.md

│── LICENSE
```

---

# 🚀 Getting Started

## Hardware Setup

- Assemble the mechanical frame.
- Install stepper motors.
- Mount belts and lead screws.
- Install extruder assembly.
- Connect Arduino Mega.
- Install RAMPS 1.4.
- Connect stepper motors.
- Install end stops.
- Connect BLTouch.
- Connect LCD display.
- Connect power supply.
- Upload Marlin Firmware.
- Perform printer calibration.

---

## Software Installation

Install Python packages

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
```

Clone Repository

```bash
git clone https://github.com/yourusername/3d-printer-with-ai-fault-detection.git

cd 3d-printer-with-ai-fault-detection
```

---

# 📊 Results

- Successfully built a fully functional 3D printer.
- Achieved excellent print accuracy.
- Total hardware cost maintained below ₹5,000.
- Successfully configured Marlin Firmware.
- Successfully integrated BLTouch automatic bed leveling.
- Successfully fine-tuned YOLOv8 for print fault detection.
- Real-time monitoring improves print reliability and reduces material waste.

---

# 🎓 Skills Gained

- Embedded Systems
- Electronics Hardware Design
- Arduino Programming
- Marlin Firmware Configuration
- Motion Control Systems
- Stepper Motor Control
- Mechanical Assembly
- 3D Printing Technology
- Computer Vision
- Deep Learning
- YOLOv8
- OpenCV
- Python Programming
- System Integration
- Hardware Troubleshooting

---

# 🔮 Future Improvements

- Wireless printer control
- OctoPrint integration
- ESP32-based remote monitoring
- Automatic print pause when defects are detected
- Mobile application support
- Multi-camera monitoring
- Cloud-based print analytics
- Predictive maintenance using AI

---

# 🤝 Contributing

Contributions are always welcome.

If you would like to improve this project:

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Yashwanth P**

Electronics and Communication Engineering

Interests

- Embedded Systems
- Hardware Design
- Firmware Development
- Computer Vision
- Artificial Intelligence
- 3D Printing
- Robotics

---

## ⭐ Support

If you found this project useful or interesting, consider giving it a **⭐ Star** on GitHub. It helps others discover the project and motivates further development.
