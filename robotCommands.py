import serial
import time
import myOwnLibrary as myJazz

commands = {
    'draw': 's215000 65000'.encode(),
    'hand0': 's185000 30000'.encode(),
    'hand1': 's260000 100000'.encode(),
    'rotator': 's170000 56000'.encode(),
    'player': 's215000 110000'.encode(),
    'zd': 'zd'.encode(),
    'zs': 'zs'.encode(),
    'zu': 'zu'.encode(),
    'ps': 'ps'.encode(),
    'pp': 'pp'.encode(),
    'v' : 'v'.encode()
}
serialPort = '/dev/tty.usbserial-0001'
pileTop = 6#s
pileBot = 8#s
pileHeight = 38#cards

def init():
    print('initializing...')
    ser = serial.Serial(serialPort, 115200, timeout=1)
    ser.write(commands['zu'])
    ser.write(commands['pp'])
    time.sleep(5)
    ser.write(commands['draw'])
    time.sleep(5)
    print('initializing done')
    ser.close()

def drawPlayer():
    ser = serial.Serial(serialPort, 115200, timeout=1)

    ser.write(commands['ps'])
    time.sleep(1)
    ser.write(commands['zd'])
    waitTime = myJazz.vectorNormalise(pileHeight, 0, 38, pileBot, pileTop)
    time.sleep(waitTime)
    ser.write(commands['zu'])
    time.sleep(waitTime)
    ser.write(commands['player'])
    time.sleep(4)
    ser.write(commands['pp'])
    time.sleep(1)
    ser.write(commands['v'])
    time.sleep(1)
    ser.write(commands['draw'])
    time.sleep(4)

    ser.close()

def drawRobot(hand):
    ser = serial.Serial(serialPort, 115200, timeout=1)

    ser.write(commands['ps'])
    time.sleep(1)
    ser.write(commands['zd'])
    waitTime = myJazz.vectorNormalise(pileHeight, 0, 38, pileBot, pileTop)
    time.sleep(waitTime)
    ser.write(commands['zu'])
    time.sleep(waitTime)
    ser.write(commands[f'hand{hand}'])
    time.sleep(3)
    ser.write(commands['pp'])
    time.sleep(1)
    ser.write(commands['v'])
    time.sleep(1)
    ser.write(commands['draw'])
    time.sleep(3)

    ser.close()

def shuffle(hand):
    ser = serial.Serial(serialPort, 115200, timeout=1)

    ser.write(commands[f'hand{hand}'])
    time.sleep(4)
    ser.write(commands['ps'])
    time.sleep(1)
    ser.write(commands['zd'])
    time.sleep(8)
    ser.write(commands['zu'])
    time.sleep(8)
    ser.write(commands[f'hand{(hand+1)%2}'])
    time.sleep(5)
    ser.write(commands['pp'])
    time.sleep(1)
    ser.write(commands['v'])
    time.sleep(1)
    ser.write(commands['draw'])
    time.sleep(4)

    ser.close()

def play(hand):
    ser = serial.Serial(serialPort, 115200, timeout=1)

    ser.write(commands[f'hand{hand}'])
    time.sleep(3)
    ser.write(commands['ps'])
    time.sleep(1)
    ser.write(commands['zd'])
    time.sleep(8)
    ser.write(commands['zu'])
    time.sleep(8)
    ser.write(commands['player'])
    time.sleep(6)
    ser.write(commands['pp'])
    time.sleep(1)
    ser.write(commands['v'])
    time.sleep(1)
    ser.write(commands['draw'])
    time.sleep(4)

    ser.close()