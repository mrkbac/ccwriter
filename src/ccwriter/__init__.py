"""CloudCompare BIN file writer.

https://www.cloudcompare.org/doc/wiki/index.php/BIN
"""

from __future__ import annotations

import io
from enum import IntFlag
from pathlib import Path
from typing import IO, Union

import numpy as np


class CCFlags(IntFlag):
    always_on = 1
    colors = 2
    normals = 4
    scalar = 8
    cloud_name = 16


class CCWriter:
    """Write a CloudCompare BIN file."""

    def __init__(self, file: Union[str, Path, IO]) -> None:
        """

        Args:
        file: str or Path or IO
            The file to write to.
        """

        self.cloud_counter = 0
        self.file = None

        if isinstance(file, (str, Path)):
            self.file = open(file, "wb")
        elif hasattr(file, "write") and callable(file.write):
            self.file = file
        else:
            raise TypeError("Expected str, Path or IO for 'file'")

        # cloud count, updated in finish()
        self.file.write((0).to_bytes(4, "little"))

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.finish()

    def add_cloud(
        self,
        cloud: np.ndarray,
        *,
        color: np.ndarray | None = None,
        normal: np.ndarray | None = None,
        scalar: np.ndarray | int | None = None,
        name: str | None = None,
    ) -> None:
        """Add a cloud to the file.

        Args:
        cloud : np.ndarray
            The cloud to add.
        colors : np.ndarray, optional
            The colors of the cloud.
        normals : np.ndarray, optional
            The normals of the cloud.
        scalar : np.ndarray | int, optional
            The scalar of the cloud, or the index in `cloud`.
        name : str, optional
            The name of the cloud.
        """
        if self.file is None:
            return

        if not isinstance(cloud, np.ndarray):
            raise TypeError("Expected np.ndarray for 'cloud'")

        if cloud.ndim != 2:
            raise ValueError("Expected 2D array for 'cloud'")
        if cloud.shape[1] < 3:
            raise ValueError("Expected at least 3 columns for 'cloud'")

        count = cloud.shape[0]
        if count > 4_294_967_296:
            raise ValueError("Expected count <= 4294967296 for 'cloud'")

        # write to buffer to prevent corruption, on exception
        buffer = io.BytesIO()
        # Number of points
        buffer.write(count.to_bytes(4, "little"))

        combined_cloud = [cloud[:, 0], cloud[:, 1], cloud[:, 2]]
        dtypes = [("x", np.float32), ("y", np.float32), ("z", np.float32)]

        flags = CCFlags.always_on
        if color is not None:
            if not isinstance(color, np.ndarray):
                raise TypeError("Expected np.ndarray for 'colors'")
            if color.shape != (count, 3):
                raise ValueError("Expected (count, 3) for 'colors'")

            combined_cloud.extend([color[:, 0], color[:, 1], color[:, 2]])
            dtypes.extend([("r", np.uint8), ("g", np.uint8), ("b", np.uint8)])

            flags |= CCFlags.colors
        if normal is not None:
            if not isinstance(normal, np.ndarray):
                raise TypeError("Expected np.ndarray for 'normals'")

            if normal.shape != (count, 3):
                raise ValueError("Expected (count, 3) for 'normals'")

            combined_cloud.extend([normal[:, 0], normal[:, 1], normal[:, 2]])
            dtypes.extend([("nx", np.float32), ("ny", np.float32), ("nz", np.float32)])

            flags |= CCFlags.normals
        if scalar is not None:
            if isinstance(scalar, int):
                if scalar < 0 or cloud.ndim >= scalar:
                    raise ValueError("Expected scalar >= 0 and < cloud.ndim for 'scalar'")
                scalar = cloud[:, scalar]

            if not isinstance(scalar, np.ndarray):
                raise TypeError("Expected np.ndarray for 'scalar'")

            if scalar.shape != (count,):
                raise ValueError("Expected (count,) for 'scalar'")

            combined_cloud.append(scalar)
            dtypes.append(("scalar", np.float64))

            flags |= CCFlags.scalar
        if name is not None:
            if not isinstance(name, str):
                raise TypeError("Expected str for 'name'")
            flags += 16

        buffer.write(flags.to_bytes(1, "little"))

        if name is not None:
            buffer.write(bytes(name, "ascii") + b"\x00")

        w_cloud = np.rec.fromarrays(combined_cloud, dtype=dtypes)

        self.cloud_counter += 1
        self.file.write(buffer.getvalue())
        self.file.write(w_cloud.tobytes())

    def finish(self) -> None:
        """Finish writing the file.

        This will update the number of clouds in the file and close the file.
        """

        if self.file is None:
            return

        self.file.seek(0)
        # Number of clouds
        self.file.write(self.cloud_counter.to_bytes(4, "little"))  # u32

        self.file.close()
        self.file = None


class CCReader(dict):
    def __init__(self, file: Union[str, Path, IO]) -> None:
        """

        Args:
        file: str or Path or IO
            The file to write to.
        """

        if isinstance(file, (str, Path)):
            file = open(file, "rb")
        elif not hasattr(file, "read") or not callable(file.read):
            raise TypeError("Expected str, Path or IO for 'file'")

        cloud_count = int.from_bytes(file.read(4), byteorder="little")  # cloud count

        for idx in range(cloud_count):
            number_of_points = int.from_bytes(file.read(4), byteorder="little")  # u32
            flags = int.from_bytes(file.read(1), byteorder="little")  # u8

            dtype = [("x", np.float32), ("y", np.float32), ("z", np.float32)]

            if flags & CCFlags.colors:
                dtype.extend([("r", np.uint8), ("g", np.uint8), ("b", np.uint8)])
            if flags & CCFlags.normals:
                dtype.extend([("nx", np.float32), ("ny", np.float32), ("nz", np.float32)])
            if flags & CCFlags.scalar:
                dtype.append(("scalar", np.float64))

            name = None
            if flags & CCFlags.cloud_name:
                name = ""
                while True:
                    char = file.read(1)
                    if char == b"\x00":
                        break
                    name += char.decode("ascii")

            cloud = np.fromfile(file, dtype=dtype, count=number_of_points)

            self[idx] = cloud
            if name is not None:
                self[name] = cloud

        file.close()
