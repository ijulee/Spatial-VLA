"""
Wrapper around the serial connection to the bluetooth robot. You must pair with
the robot first, check the correct outgoing COM port number in Device Manager, 
and ***REPLACE THE PORT NAME VARIABLE*** before using this class to open the
serial connection. 

 * Bluetooth device name: SVLM-bot
 * Bluetooth device address: 00:14:03:06:75:c2
 * Bluetooth connection PIN: 1234

"""
import time
import serial

PORT = 'COM4' # REPLACE WITH CORRECT OUTGOING PORT NUMBER
BAUD_RATE = 9600

class BluetoothBot:
    ser = None

    # establish the serial connection
    def open_connection(self):
        """establish the serial connection"""
        try:
            print(f"Connecting to {PORT}...")
            self.ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
            time.sleep(2) # Stabilize connection
            print(f"Connected to {PORT}!")

        except serial.SerialException as e:
            print(f"Could not connect to {PORT}: {e}. Check Bluetooth settings.")
        finally:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
                print("Connection closed.")

    # close the serial connection
    def close_connection(self):
        """close the serial connection"""
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
                print("Connection closed.")
                self.ser = None
            else: 
                print("No open serial connection")
        except serial.SerialException as e:
            print(f"Error: {e}")

    # send message through serial connection
    def send_message(self, message):
        """send message through serial connection"""
        try: 
            if self.ser.is_open:
                outgoing = str(message) + "\n"
                self.ser.write(outgoing.encode('utf-8'))
            else:
                print("No open serial connection")
        except serial.SerialException as e:
            print(f"Error: {e}")
