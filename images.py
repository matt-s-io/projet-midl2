import numpy as np
import struct


def load_mnist_idx(file_path):
  with open(file_path, "rb") as f:
    (magic,) = struct.unpack(">I", f.read(4))

    if magic == 2051:
      num, rows, cols = struct.unpack(">III", f.read(12))
      data = np.frombuffer(f.read(), dtype=np.uint8)
      return data.reshape(num, rows * cols)

    elif magic == 2049:  # Fichier de labels (.idx1)
      (num,) = struct.unpack(">I", f.read(4))
      data = np.frombuffer(f.read(), dtype=np.uint8)
      return data

    else:
      raise ValueError(f"Format IDX non reconnu. Magic number: {magic}")
