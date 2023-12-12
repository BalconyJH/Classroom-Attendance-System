import asyncio
from pathlib import Path

from src.utils.camera import Camera
from src.utils.face_classifier import FaceClassifier
from src.utils import logger, return_euclidean_distance

path = Path(__file__).parent / "resources" / "caches" / "dataset" / "images"


async def extract_face_descriptor():
    FaceClassifier().preprocess_face(str(path / "1.jpg"))
    test_face_descriptors = await FaceClassifier().face_descriptors(str(path / "1.jpg"))
    print(test_face_descriptors)


async def test_compare():
    face1_numpy_array = await FaceClassifier().face_descriptors(str(path / "test" / "1.jpg"))
    face2_numpy_array = await FaceClassifier().face_descriptors(str(path / "unk" / "1.jpg"))

    if return_euclidean_distance(face1_numpy_array, face2_numpy_array):
        logger.info("两张人脸相似")
    else:
        logger.info("两张人脸不相似")

async def main1():
    for image in path.iterdir():
        FaceClassifier().preprocess_face(str(image))
        # with open("test.pkl", "rb") as f:
        #     oc_svm = pickle.load(f)
        # result = is_same_person(oc_svm, test_face_descriptors[0])
        # print(result)
    # test_face_descriptors1 = FaceClassifier().detect_face_landmarks(str(path))
    # oc_svm = OneClassSVM(gamma='auto')
    # oc_svm.fit(test_face_descriptors1)
    #
    # test_face_descriptors = await FaceClassifier().face_descriptors(str(path / "13.jpg"))
    # print(test_face_descriptors)
    # result = is_same_person(oc_svm, test_face_descriptors[0])
    # print(result)


async def main2():
    await Camera("F:\\Classroom-Attendance-System\\src\\resources\\statics\\unknow.mp4").face_gather(uid="unk")


if __name__ == "__main__":
    asyncio.run(test_compare())
