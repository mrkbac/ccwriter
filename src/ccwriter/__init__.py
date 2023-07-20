import io
from pathlib import Path
from typing import IO

import numpy as np


class CCWriter:
    """
    https://www.cloudcompare.org/doc/wiki/index.php/BIN
    """

    def __init__(self, file: str | Path | IO, *, write: bool = True) -> None:
        self.cloud_counter = 0
        self.file = None
        if write:
            if isinstance(file, str | Path):
                self.file = open(file, "wb")  # noqa: SIM115, PTH123
            else:
                self.file = file
            # cloud count, updated in finish()
            self.file.write(np.array(0, dtype=np.uint32).tobytes())

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.finish()

    def add_cloud(
        self,
        cloud: np.ndarray,
        *,
        colors: np.ndarray | None = None,
        normals: np.ndarray | None = None,
        scalar: np.ndarray | int | None = None,
        name: str | None = None,
    ) -> None:
        """
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

        # write to buffer to prevent corruption, on exception
        buffer = io.ByteIO()
        # Number of points
        buffer.write(np.array(count, dtype=np.uint32).tobytes())


        combined_cloud = [cloud[:, 0], cloud[:, 1], cloud[:, 2]]
        dtypes = [("x", np.float32), ("y", np.float32), ("z", np.float32)]

        flags = 1
        if colors is not None:
            if not isinstance(colors, np.ndarray):
                raise TypeError("Expected np.ndarray for 'colors'")
            if colors.shape != (count, 3):
                raise ValueError("Expected (count, 3) for 'colors'")

            combined_cloud.extend([colors[:, 0], colors[:, 1], colors[:, 2]])
            dtypes.extend([("r", np.uint8), ("g", np.uint8), ("b", np.uint8)])

            flags += 2
        if normals is not None:
            if not isinstance(normals, np.ndarray):
                raise TypeError("Expected np.ndarray for 'normals'")

            if normals.shape != (count, 3):
                raise ValueError("Expected (count, 3) for 'normals'")

            combined_cloud.extend([normals[:, 0], normals[:, 1], normals[:, 2]])
            dtypes.extend([("nx", np.float32), ("ny", np.float32), ("nz", np.float32)])

            flags += 4
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

            flags += 8
        if name is not None:
            if not isinstance(name, str):
                raise TypeError("Expected str for 'name'")
            flags += 16

        buffer.write(np.array(flags, dtype=np.uint8).tobytes())

        if name is not None:
            buffer.write(bytes(name, "ascii") + b"\x00")

        w_cloud = np.rec.fromarrays(combined_cloud, dtype=dtypes)


        self.cloud_counter += 1
        self.file.write(buffer.getvalue())
        self.file.write(w_cloud.tobytes())


    def finish(self) -> None:
        if self.file is None:
            return

        self.file.seek(0)
        # Number of clouds
        self.file.write(np.array(self.cloud_counter, dtype=np.uint32).tobytes())

        self.file.close()
        self.file = None
