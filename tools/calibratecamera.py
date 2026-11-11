import cv2 as cv
import numpy as np
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

    calibrationOutput: str = "camera_calibration.yaml"

    @model_validator(mode="after")
    def validate_script(self):
        if self.calibrationDir[-1] != "/":
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


# Paths in config.toml are relative to this script's directory
SCRIPT_DIR = str(Path(__file__).resolve().parent)

# Fewer corners than this per image gives an unreliable pose estimate
MIN_CORNERS_PER_IMAGE = 6


def load_config(
    config_path=os.path.join(SCRIPT_DIR, "config.toml"),
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

        # gridColumns/gridRows count squares; OpenCV wants the inner corners
        pattern_size = (config.gridColumns - 1, config.gridRows - 1)

        object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        object_points[:, :2] = (
            np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
            * config.gridSquareSize
        )

        # No detector object exists for plain chessboards, so the "board" is the
        # pattern size plus its 3D corner layout and the detector is None
        return (pattern_size, object_points), None

    raise ValueError(f"Unsupported calibration board: {config.type}")


def detect_board(gray, board, detector):
    """Return matching (object_points, image_points) for one image, or (None, None)."""

    if detector is not None:

        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        if charuco_ids is None or len(charuco_ids) < MIN_CORNERS_PER_IMAGE:
            return None, None

        object_points, image_points = board.matchImagePoints(
            charuco_corners, charuco_ids
        )
        return object_points, image_points

    pattern_size, object_points = board
    found, corners = cv.findChessboardCornersSB(gray, pattern_size)

    if not found:
        return None, None

    return object_points, corners


def capture_live_calibration_data(board, detector, script_config, camera_config):
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
    imagePath = SCRIPT_DIR + calibration_image_path + script_config.imageFilePrefix
    os.makedirs(SCRIPT_DIR + calibration_image_path, exist_ok=True)

    print("Press C to capture an image, Q to finish capturing.")

    # Start camera preview
    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame.")
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Read the keyboard once per frame so key presses aren't dropped
            key = cv.waitKey(1) & 0xFF

            if key == ord("c"):

                file_name = f"{imagePath}{count}{datetime.now()}.png"
                cv.imwrite(file_name, frame)
                count += 1
                print("Image Written at: " + file_name)

            # Press Q to exit
            if key == ord("q"):
                break

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

            else:

                pattern_size, _ = board
                found, corners = cv.findChessboardCornersSB(gray, pattern_size)
                cv.drawChessboardCorners(frame, pattern_size, corners, found)

            cv.imshow("Camera Calibration", frame)

    finally:
        cap.release()
        cv.destroyAllWindows()


def generate_calibration_board_output(board, board_config):
    if board_config.type != "Charuco":
        print("Board generation is only supported for ChArUco boards, skipping.")
        return

    # Create Board Image
    board_image_size = (2000, 1500)
    board_image = board.generateImage(board_image_size)
    cv.imwrite(
        os.path.join(
            SCRIPT_DIR,
            f"{board_config.gridColumns}x{board_config.gridRows}_Charuco_Board.png",
        ),
        board_image,
    )


def process_images(ImageFolderPath, board, detector):
    """Detect the board in every image and collect matching 3D/2D point sets."""
    objectPoints = []
    imagePoints = []
    usedFiles = []
    imageSize = None
    folder = SCRIPT_DIR + ImageFolderPath

    for file in sorted(os.listdir(folder)):
        if not (file.endswith(".jpg") or file.endswith(".png")):
            continue

        img = cv.imread(folder + file)
        if img is None:
            print(f"Could not read {file}, skipping.")
            continue

        size = (img.shape[1], img.shape[0])
        if imageSize is None:
            imageSize = size
        elif size != imageSize:
            print(f"{file} is {size}, expected {imageSize}, skipping.")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        object_points, image_points = detect_board(gray, board, detector)

        if object_points is None or len(object_points) < MIN_CORNERS_PER_IMAGE:
            print(f"Board not found (or too few corners) in {file}, skipping.")
            continue

        objectPoints.append(object_points)
        imagePoints.append(image_points)
        usedFiles.append(file)

    return objectPoints, imagePoints, usedFiles, imageSize


def run_calibration(board, detector, script_config, camera_config):
    objectPoints, imagePoints, usedFiles, imageSize = process_images(
        script_config.calibrationDir, board, detector
    )

    if len(usedFiles) < 3:
        raise RuntimeError(
            f"Only {len(usedFiles)} usable calibration images, need at least 3 "
            "(15-30 from varied angles is recommended)."
        )

    # The camera may not honour the requested resolution, so calibrate at the
    # size the images were actually captured at
    if imageSize != tuple(camera_config.resolution):
        print(
            f"Warning: images are {imageSize[0]} x {imageSize[1]} but config resolution "
            f"is {camera_config.resolution[0]} x {camera_config.resolution[1]}. "
            "Calibrating at the image size."
        )

    print(f"Calibrating with {len(usedFiles)} images at {imageSize[0]} x {imageSize[1]}.")

    # calibrateCameraRO Parameters
    # iFixedPoint = -1 disables the release-object method, matching calibrateCamera
    iFixedPoint = -1
    # CameraMatrix parameter must be None if you do not have an initial guess of the camera matrix
    flags = 0
    criteria = (cv.TERM_CRITERIA_COUNT + cv.TERM_CRITERIA_EPS, 100, 1e-6)

    rms, cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints = cv.calibrateCameraRO(
        objectPoints,
        imagePoints,
        imageSize,
        iFixedPoint,
        None,
        None,
        flags=flags,
        criteria=criteria,
    )

    # Per image error makes it easy to spot and delete bad captures
    print("Per image reprojection error (px):")
    for file, obj, img, rvec, tvec in zip(usedFiles, objectPoints, imagePoints, rvecs, tvecs):
        projected, _ = cv.projectPoints(obj, rvec, tvec, cameraMatrix, distCoeffs)
        error = np.sqrt(np.mean(np.sum((img.reshape(-1, 2) - projected.reshape(-1, 2)) ** 2, axis=1)))
        flag = "  <-- consider removing" if error > 1.0 else ""
        print(f"  {file:40s} {error:.3f}{flag}")

    print(f"RMS reprojection error: {rms:.4f} px")
    print(f"Camera matrix:\n{cameraMatrix}")
    print(f"Distortion coefficients: {distCoeffs.ravel()}")

    save_calibration(
        os.path.join(SCRIPT_DIR, script_config.calibrationOutput),
        imageSize,
        cameraMatrix,
        distCoeffs,
        rms,
    )

    return cameraMatrix, distCoeffs


def save_calibration(output_path, imageSize, cameraMatrix, distCoeffs, rms):
    # OpenCV YAML so the result can be read back from Python or C++ with cv.FileStorage
    storage = cv.FileStorage(output_path, cv.FILE_STORAGE_WRITE)
    storage.write("image_width", imageSize[0])
    storage.write("image_height", imageSize[1])
    storage.write("camera_matrix", cameraMatrix)
    storage.write("distortion_coefficients", distCoeffs)
    storage.write("rms_reprojection_error", rms)
    storage.release()
    print(f"Calibration saved to {output_path}")


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
        capture_live_calibration_data(board, detector, script_config, camera_config)
    run_calibration(board, detector, script_config, camera_config)


if __name__ == "__main__":
    main()
