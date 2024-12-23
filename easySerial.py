import serial

serialPort = '/dev/ttyUSB0'


while True:
    command = input()
    ser = serial.Serial(serialPort, 115200)
    ser.write(command.encode())
    ser.close()