import serial
import time
import myOwnLibrary as myJazz

commands = {
    'draw': 's210000 60000',
    'hand1': 's240000 100000',
    'hand2': 's220000 110000',
    'discard': 's185000 85000',
    'playerDeal': 's200000 120000',
    'zd': 'zd',
    'zs': 'zs',
    'zu': 'zu',
}
travelTime = 2
pickUpHands = 6
dropHands = 4

suctionTime = 3
serialPort = '/dev/ttyUSB0'

pileHeight = 64

def moveTo(location):
    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zu'].encode())
    ser.close()
    time.sleep(5)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands[location].encode())
    ser.close()
    time.sleep(travelTime)

def init():
    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zu'].encode())
    ser.close()
    time.sleep(5)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['draw'].encode())
    ser.close()
    time.sleep(travelTime)

def pickUpHand1():
    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['hand1'].encode())
    ser.close()
    time.sleep(travelTime)
    
    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zd'].encode())
    ser.close()
    time.sleep(pickUpHands)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zs'].encode())
    ser.close()
    time.sleep(suctionTime)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zu'].encode())
    ser.close()
    time.sleep(1)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zs'].encode())
    ser.close()
    time.sleep(suctionTime)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zu'].encode())
    ser.close()
    time.sleep(pickUpHands+1)

def pickUpHand2():
    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['hand2'].encode())
    ser.close()
    time.sleep(travelTime)
    
    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zd'].encode())
    ser.close()
    time.sleep(pickUpHands)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zs'].encode())
    ser.close()
    time.sleep(suctionTime)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zu'].encode())
    ser.close()
    time.sleep(1)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zs'].encode())
    ser.close()
    time.sleep(suctionTime)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zu'].encode())
    ser.close()
    time.sleep(pickUpHands+1)

def dropHand1():
    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['hand1'].encode())
    ser.close()
    time.sleep(travelTime)
    
    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zd'].encode())
    ser.close()
    time.sleep(dropHands)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zs'].encode())
    ser.close()
    time.sleep(suctionTime)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zu'].encode())
    ser.close()
    time.sleep(dropHands)

def dropHand2():
    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['hand2'].encode())
    ser.close()
    time.sleep(travelTime)
    
    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zd'].encode())
    ser.close()
    time.sleep(dropHands)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zs'].encode())
    ser.close()
    time.sleep(suctionTime)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zu'].encode())
    ser.close()
    time.sleep(dropHands)

def drawCard():
    global pileHeight
    pickTime = myJazz.vectorNormalise(pileHeight, 1, 98, 7.867078, 2.416013)
    pileHeight -=1
    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['draw'].encode())
    ser.close()
    time.sleep(travelTime)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zd'].encode())
    ser.close()
    time.sleep(pickTime)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zs'].encode())
    ser.close()
    time.sleep(suctionTime)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zu'].encode())
    ser.close()
    time.sleep(2)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zs'].encode())
    ser.close()
    time.sleep(5)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zu'].encode())
    ser.close()
    time.sleep(abs(pickTime - 1)+2)

def playCard():
    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['discard'].encode())
    ser.close()
    time.sleep(travelTime)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zd'].encode())
    ser.close()
    time.sleep(dropHands)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zs'].encode())
    ser.close()
    time.sleep(suctionTime)

    ser = serial.Serial(serialPort, 115200)
    ser.write(commands['zu'].encode())
    ser.close()
    time.sleep(dropHands)