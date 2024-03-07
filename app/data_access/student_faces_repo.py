from app.database.models import Faces


class StudentFaces:
    @staticmethod
    def get_all_student_faces() -> list[Faces]:
        """
        获取所有学生人脸特征
        :return: list[Faces]
        """
        return Faces.query.all()
