import os

if __name__ == '__main__':
    for (_, _, files) in os.walk("./input"):
        for file in files:
            if file.endswith(".png"):
                os.system(f"python -m p1.main ./input/{file} --show")
