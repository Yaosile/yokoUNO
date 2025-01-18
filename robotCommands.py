import serial
import time
import myOwnLibrary as myJazz

commands = {
    'draw': 's215000 65000'.encode(),
    'hand0': 's175000 30000'.encode(),
    'hand1': 's260000 100000'.encode(),
    'rotator': 's215000 95000'.encode(),
    'player': 's170000 55000'.encode(),
    'zd': 'zd'.encode(),
    'zs': 'zs'.encode(),
    'zu': 'zu'.encode(),
    'ps': 'ps'.encode(),
    'pp': 'pp'.encode(),
    'v' : 'v'.encode(),
    'flip': 'r300000'.encode(),
    'play': 'r-300000'.encode(),
}

serialPort = '/dev/tty.usbserial-0001'
pileTop = 6#s
pileBot = 8#s
pileHeight = 38#cards

def init():
    print('initialising...')
    ser = serial.Serial(serialPort, 115200, timeout=1)
    ser.write(commands['zu'])
    time.sleep(1)
    ser.write(commands['pp'])
    time.sleep(5)
    ser.write(commands['draw'])
    time.sleep(5)
    print('initialising done')
    ser.close()

def drawPlayer():
    global pileHeight
    ser = serial.Serial(serialPort, 115200, timeout=1)

    ser.write(commands['ps'])
    time.sleep(1)
    waitTime = myJazz.vectorNormalise(pileHeight, 0, 38, pileBot, pileTop)
    ser.write(commands['zd'])
    pileHeight -= 1
    time.sleep(waitTime)
    ser.write(commands['zu'])
    time.sleep(waitTime)
    ser.write(commands['player'])
    time.sleep(4)
    ser.write(commands['pp'])
    time.sleep(1)
    ser.write(commands['v'])
    time.sleep(1.5)
    ser.write(commands['draw'])
    time.sleep(4)

    ser.close()

def drawRobot(hand):
    global pileHeight
    ser = serial.Serial(serialPort, 115200, timeout=1)

    ser.write(commands['ps'])
    time.sleep(1)
    waitTime = myJazz.vectorNormalise(pileHeight, 0, 38, pileBot, pileTop)
    ser.write(commands['zd'])
    print(waitTime)
    pileHeight -= 1
    time.sleep(waitTime)
    ser.write(commands['zu'])
    time.sleep(waitTime)
    ser.write(commands[f'hand{hand}'])
    time.sleep(3)
    ser.write(commands['pp'])
    time.sleep(1)
    ser.write(commands['v'])
    time.sleep(1.5)
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
    time.sleep(1.5)
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
    ser.write(commands['draw'])
    time.sleep(3)
    ser.write(commands['rotator'])
    time.sleep(4)
    ser.write(commands['pp'])
    time.sleep(1)
    ser.write(commands['zd'])
    time.sleep(1)
    ser.write(commands['zs'])
    time.sleep(1)
    ser.write(commands['v'])
    time.sleep(1.5)
    ser.write(commands['zu'])
    time.sleep(2)
    ser.write(commands['draw'])
    time.sleep(3)
    ser.write(commands['flip'])
    time.sleep(6)
    ser.write(commands['play'])
    time.sleep(6)
    ser.close()