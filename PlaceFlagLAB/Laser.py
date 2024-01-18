import serial.tools.list_ports
import time

# Define the COM port and baud rate based on your Arduino configuration
arduino_port = 'COM5'  # Replace 'x' with your actual COM port
baud_rate = 115200


def send_time(t):
    

    # Open a serial connection to the Arduino
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    data = str(t) + "\n"
    print(data)
    ser.write(bytes(data, 'utf-8'))
    while 1:
        print(ser.read())


# send_time(1235)

# send_time(2000)

