from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.config import config
from src.utils import (
    init,
    database_init,
    message_translator,
    return_euclidean_distance,
    init_shape_predictor_model_file,
    init_face_recognition_model_file,
)


@patch("src.utils.database_init")
@patch("src.utils.init_shape_predictor_model_file")
@patch("src.utils.init_face_recognition_model_file")
async def test_init_with_successful_initializations(
    mock_face_recognition_init, mock_shape_predictor_init, mock_db_init
):
    await init()
    mock_db_init.assert_called_once()
    mock_shape_predictor_init.assert_called_once()
    mock_face_recognition_init.assert_called_once()


@pytest.mark.asyncio
async def test_init_with_database_initialization_failure():
    with patch("src.database.db.create_db", side_effect=SQLAlchemyError):
        with patch("src.utils.logger.error") as mock_logger_error:
            await database_init()
            mock_logger_error.assert_called_once_with(message_translator("ERRORS.DATABASE.INIT.FAILED"))


@patch("src.utils.unzip_static", side_effect=lambda x, y: True)
def test_unzip_static_with_valid_path(mock_unzip_static):
    from src.utils import unzip_static

    assert unzip_static("valid_path", "valid_path") is True
    mock_unzip_static.assert_called_once_with("valid_path", "valid_path")


@patch("src.utils.unzip_static", return_value=False)
def test_unzip_static_with_invalid_path(mock_unzip_static):
    from src.utils import unzip_static

    assert unzip_static("invalid_path", "invalid_path") is False
    mock_unzip_static.assert_called_once_with("invalid_path", "invalid_path")


def test_return_euclidean_distance_with_similar_features():
    assert return_euclidean_distance([1.0, 1.0], [1.0, 1.0]) is True


def test_return_euclidean_distance_with_dissimilar_features():
    assert return_euclidean_distance([1.0, 1.0], [2.0, 2.0]) is False


@pytest.mark.asyncio
async def test_init_shape_predictor_model_file_with_successful_initialization():
    with patch("dlib.shape_predictor", return_value=MagicMock()) as mock_shape_predictor:
        await init_shape_predictor_model_file()
        mock_shape_predictor.assert_called_once_with(f"{config.face_model_path}/shape_predictor_68_face_landmarks.dat")


@pytest.mark.asyncio
async def test_init_shape_predictor_model_file_with_initialization_failure():
    with patch("dlib.shape_predictor", side_effect=Exception()), patch("src.utils.unzip_static") as mock_unzip_static:
        await init_shape_predictor_model_file()
        mock_unzip_static.assert_called_once_with(
            config.static_path / "shape_predictor_68_face_landmarks.dat.bz2",
            config.face_model_path / "shape_predictor_68_face_landmarks.dat",
        )


@pytest.mark.asyncio
async def test_init_face_recognition_model_file_with_successful_initialization():
    with patch("dlib.face_recognition_model_v1", return_value=MagicMock()) as mock_face_recognition_model:
        await init_face_recognition_model_file()
        mock_face_recognition_model.assert_called_once_with(
            f"{config.face_model_path}/dlib_face_recognition_resnet_model_v1.dat"
        )


@pytest.mark.asyncio
async def test_init_face_recognition_model_file_with_initialization_failure():
    with patch("dlib.face_recognition_model_v1", side_effect=Exception()), patch(
        "src.utils.unzip_static"
    ) as mock_unzip_static:
        await init_face_recognition_model_file()
        mock_unzip_static.assert_called_once_with(
            config.static_path / "dlib_face_recognition_resnet_model_v1.dat.bz2",
            config.face_model_path / "dlib_face_recognition_resnet_model_v1.dat",
        )
