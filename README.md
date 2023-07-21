# CloudCompare bin writer

Writer for the CloudCompare [BIN Format](https://www.cloudcompare.org/doc/wiki/index.php/BIN)

## Example

```python
import numpy as np

from ccwriter import CCWriter

amount = 10_000

cloud = np.random.uniform(-100, 100, (amount, 3))
color = np.random.uniform(-100, 100, (amount, 3))
normal = np.random.uniform(-100, 100, (amount, 3))
scaler = np.random.uniform(-100, 100, amount)

cloud_scaler = np.random.uniform(-100, 100, (amount, 4))

with CCWriter("cloud.bin") as cc:
    cc.add_cloud(cloud, name="cloud")
    cc.add_cloud(cloud, name="cloud & color", color=color)
    cc.add_cloud(cloud, name="cloud & color & normal", color=color, normal=normal)
    cc.add_cloud(cloud, name="cloud & color & normal & scaler", color=color, normal=normal, scalar=scaler)

    cc.add_cloud(cloud_scaler, name="cloud_scaler & scaler_index", scalar=3)

```
