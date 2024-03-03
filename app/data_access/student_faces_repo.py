from app.database.models import Faces


class StudentFaces:
    @staticmethod
    async def get_all_student_faces() -> list[Faces]:
        return Faces.query.all()
