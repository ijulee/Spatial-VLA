import serial
import serial.tools.list_ports
import threading
import time
import sys
from datetime import datetime

# --- CONFIGURATION ---
# Use the auto-finder or hardcode your port (e.g., PORT = 'COM4')
PORT = 'COM4' 
BAUD_RATE = 9600
    
# ---------------------

class TimeTracker:
    outTime = None
    inTime = None

def receive_data(ser, tracker):
    """
    This function runs in the background. 
    It constantly checks for data from the robot and prints it.
    """
    inTime = None

    while ser.is_open:
        try:
            # Check if there is data waiting in the buffer
            if ser.in_waiting > 0:
                # Read the line, decode it, and strip whitespace
                incoming = ser.readline().decode('utf-8', errors='ignore').strip()
                if incoming:
                    # \r clears the "Command >> " line so the message prints cleanly
                    # Then we reprint the prompt so you know you can still type
                    tracker.inTime = datetime.now()
                    print(f"\rReply: {incoming}")
                    print("Command: ", end="", flush=True)
                    
            time.sleep(0.1) # efficient waiting
            if tracker.inTime != None and tracker.outTime != None:
                timeDiff = (tracker.inTime-tracker.outTime)
                tracker.inTime = tracker.outTime = None
                print(f"Round trip time: {timeDiff.total_seconds()}")
                print("Command: ", end="", flush=True)
            
        except serial.SerialException:
            print("\n[Error] Connection lost in receiver thread.")
            break
        except OSError:
            break

def main():
    tracker = TimeTracker()
    try:
        print(f"Connecting to {PORT}...")
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Stabilize connection
        print(f"Connected to {PORT}!")
        print("Type commands below. Messages from robot will appear automatically.")
        print("Type 'exit' to quit.\n")

        # --- START THE LISTENER THREAD ---
        # daemon=True means this thread will automatically die when the main program quits
        listener = threading.Thread(target=receive_data, args=(ser, tracker), daemon=True)
        listener.start()

        # --- MAIN LOOP (SENDING COMMANDS) ---
        while True:
            # The 'end' and 'flush' are just to make the prompt look nice
            user_input = input("Command: ")

            if user_input.lower() == 'exit':
                print("Exiting...")
                break
            
            tracker.outTime = datetime.now()

            # Send the command
            if ser.is_open:
                cmd_with_newline = user_input + "\n"
                ser.write(cmd_with_newline.encode('utf-8'))

    except serial.SerialException as e:
        print(f"Could not connect to {PORT}: {e}. Check Bluetooth settings.")
    except KeyboardInterrupt:
        print("\nForce closed.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Connection closed.")

if __name__ == "__main__":
    main()