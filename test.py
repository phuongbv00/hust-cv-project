import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    input_dir = os.path.join(script_dir, "input")
    
    for (_, _, files) in os.walk(input_dir):
        for file in files:
            if file.endswith(".png"):
                # Construct the full path to the file
                file_path = os.path.join(input_dir, file)
                
                # Use a more robust way to run the command
                command = [
                    sys.executable,  # Use the same Python interpreter
                    "-m",
                    "p1.main",
                    file_path,
                    "--show"
                ]
                
                # Execute the command
                os.system(" ".join(command))