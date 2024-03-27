import logging
import sys

cmd = 1
speed = 2
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

logging.info(f"cmd: {cmd}, speed: {speed}")
