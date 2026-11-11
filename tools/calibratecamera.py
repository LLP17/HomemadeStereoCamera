import cv2 as cv
import tomllib
import os
from datetime import datetime

from pathlib import Path
from typing import Literal, Annotated

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    model_validator,
)

# ============================================================
# Configuration Models
# ============================================================


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cameraId: int = Field(ge=0)

    resolution: tuple[
        Annotated[int, Field(gt=0)],
        Annotated[int, Field(gt=0)],
    ]


class ScriptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calibrationDir: str

    calibrationLive: bool

    imageFilePrefix: str

    generateCalibrationBoard: bool

    @model_validator(mode="after")
    def validate_script(self):
        if self.calibrationDir[:-1] != "/":
            self.calibrationDir = self.calibrationDir + "/"
        return self


class CalibrationBoardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["Charuco", "Chessboard"]

    gridColumns: int = Field(ge=2)
    gridRows: int = Field(ge=2)

    gridSquareSize: float = Field(gt=0)
    markerLength: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_board(self):

        if self.type == "Charuco":

            if self.markerLength is None:
                raise ValueError("ChArUco boards require a marker length.")

            if self.markerLength >= self.gridSquareSize:
                raise ValueError("Marker length must be smaller than square size.")

        return self


class ConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera: CameraConfig
    calibration_board: CalibrationBoardConfig
    script: ScriptConfig


# ============================================================
# Configuration Loader
# ============================================================


def load_config(
    config_path="/home/lawrence/Documents/Projects/HomemadeStereoCamera/tools/config.toml",
) -> ConfigSchema:

    with open(config_path, "rb") as file:
        data = tomllib.load(file)

    return ConfigSchema.model_validate(data)


# ============================================================
# Calibration Board Initialization
# ============================================================


def create_calibration_board(config: CalibrationBoardConfig):

    if config.type == "Charuco":

        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)

        board = cv.aruco.CharucoBoard(
            (config.gridColumns, config.gridRows),
            config.gridSquareSize,
            config.markerLength,
            dictionary,
        )

        detector = cv.aruco.CharucoDetector(board)

        return board, detector

    elif config.type == "Chessboard":

        # Chessboard detection can be implemented here.
        return None, None

    raise ValueError(f"Unsupported calibration board: {config.type}")


def capture_live_calibration_data(detector, script_config, camera_config):
    # Initialize camera
    cap = cv.VideoCapture(camera_config.cameraId, cv.CAP_V4L)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera {camera_config.cameraId}")

    # Configure camera resolution
    width, height = camera_config.resolution

    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    actual_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    print(f"Actual camera resolution: {actual_width} x {actual_height}")

    count = 0
    calibration_image_path = script_config.calibrationDir
    imagePath = os.getcwd() + calibration_image_path + script_config.imageFilePrefix
    os.makedirs(os.getcwd() + calibration_image_path, exist_ok=True)

    # Start camera preview
    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame.")
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if cv.waitKey(1) & 0xFF == ord("c"):

                cv.imwrite(
                    f"{imagePath}{count}{datetime.now()}.png",
                    frame,
                )
                count += 1
                print("Image Written at: " + imagePath + f"{count}")
            # Detect ChArUco corners
            if detector is not None:

                charuco_corners, charuco_ids, marker_corners, marker_ids = (
                    detector.detectBoard(gray)
                )

                if (
                    charuco_corners is not None
                    and charuco_ids is not None
                    and len(charuco_ids) > 0
                ):

                    cv.aruco.drawDetectedCornersCharuco(
                        frame,
                        charuco_corners,
                        charuco_ids,
                    )

            cv.imshow("Camera Calibration", frame)

            # Press Q to exit
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv.destroyAllWindows()


def generate_calibration_board_output(board, board_config):
    # Create Board Image
    board_image_size = (2000, 1500)
    board_image = board.generateImage(board_image_size)
    cv.imwrite(
        f"{board_config.gridColumns}x{board_config.gridRows}_Charuco_Board.png",
        board_image,
    )


def process_images(ImageFolderPath, detector):
    images = []
    current_working_directory = os.getcwd()
    for file in os.listdir(current_working_directory + ImageFolderPath):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv.imread(current_working_directory + ImageFolderPath + f"{file}")
            if img is not None:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                corners, ids, markers, marker_ids = detector.detectBoard(gray)
                images.append(img)
                # print(current_working_directory + ImageFolderPath + f"{file}")
    return images


def run_calibration(detector, script_config, camera_config):
    # calibrateCameraR0 Parameters
    objectPoints = []
    imagePoints = []
    imageSize = CameraConfig.resolution
    iFixedPoint = -1
    # CameraMatrix parameter must be None if you do not have an initial guess of the camera matrix
    distCoeffs = []
    flags = None
    criteria = None
    # process_images()
    cv.calibrateCameraRO(
        objectPoints, imagePoints, imageSize, iFixedPoint, None, None, 0
    )
    pass


# ============================================================
# Main Application
# ============================================================


def main():
    # Load and validate configuration
    config = load_config()

    camera_config = config.camera
    board_config = config.calibration_board
    script_config = config.script

    print("Configuration loaded successfully.")
    print(f"Camera ID: {camera_config.cameraId}")
    print(f"Resolution: {camera_config.resolution}")
    print(f"Calibration board: {board_config.type}")

    # Initialize calibration board
    board, detector = create_calibration_board(board_config)
    if script_config.generateCalibrationBoard == True:
        generate_calibration_board_output(board, board_config)
    if script_config.calibrationLive == True:
        capture_live_calibration_data(detector, script_config, camera_config)
    run_calibration()


if __name__ == "__main__":
    main()
