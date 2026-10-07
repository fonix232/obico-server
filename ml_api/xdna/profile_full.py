import json, os, sys, time
import cv2, numpy as np
sys.path.insert(0, "/w")
os.environ["YOLO_PROFILE"] = "1"
from yolo_npu import YoloNpu
net = YoloNpu("/w/model.onnx", json.load(open("/w/scales.json")), "/cache/rt")
img = cv2.imread("/w/detached-1.jpg")
x = np.expand_dims(np.transpose(cv2.cvtColor(cv2.resize(img, (416, 416)), cv2.COLOR_BGR2RGB), (2, 0, 1)).astype(np.float32), 0) / 255.0
net.run(x)
acc = {}
cpu0, wall0 = time.process_time(), time.perf_counter()
for _ in range(10):
    net.run(x)
    for k, v in net.prof.items():
        acc[k] = acc.get(k, 0) + v / 10
print(f"per frame: wall {(time.perf_counter()-wall0)/10*1e3:.1f} ms, process CPU {(time.process_time()-cpu0)/10*1e3:.1f} ms")
npu = sum(v for k, v in acc.items() if k.startswith("npu"))
host = sum(v for k, v in acc.items() if not k.startswith("npu"))
print(f"NPU total {npu*1e3:.1f} ms, host total {host*1e3:.1f} ms")
for k, v in sorted(acc.items(), key=lambda kv: -kv[1])[:16]:
    print(f"  {k:12} {v*1e3:6.2f} ms")
