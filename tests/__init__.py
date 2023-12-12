import timeit
import unittest

from src.utils.camera import CameraEvent as DictCameraEvent
from src.utils.camera import RBTreeCameraEvent as RBTreeCameraEvent


class TestCameraEventPerformance(unittest.TestCase):
    def setUp(self):
        self.num_events = 10

    def test_dict_camera_event(self):
        event = DictCameraEvent()
        start_time = timeit.default_timer()
        for i in range(self.num_events):
            event.set()
            event.clear()
        end_time = timeit.default_timer()
        print(f"DictCameraEvent time: {end_time - start_time}")

    def test_rbtree_camera_event(self):
        event = RBTreeCameraEvent()
        start_time = timeit.default_timer()
        for i in range(self.num_events):
            event.set()
            event.clear()
        end_time = timeit.default_timer()
        print(f"RBTreeCameraEvent time: {end_time - start_time}")


if __name__ == "__main__":
    unittest.main()
