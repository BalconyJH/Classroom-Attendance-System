import asyncio

from utils.camera import Camera

if __name__ == "__main__":
    asyncio.run(Camera(0).face_detector())
