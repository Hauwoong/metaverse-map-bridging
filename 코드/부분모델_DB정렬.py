#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""합본 모델에서 경계 실촬영만 뽑아 **데이터베이스의 ID 체계로** 부분 모델을 쓴다.

🔴 왜 ID 를 다시 매기는가.
   COLMAP 의 데이터베이스는 이미지에 1번부터 새로 번호를 매긴다. 합본 모델의 원래
   번호를 그대로 쓰면 같은 번호가 서로 다른 파일을 가리켜 등록기가 죽는다:
     Check failed: existing_image.Name() == image.Name() (v1791_00143.png vs gen_0143.png)
   rig/frame 도 같은 이유로 충돌한다.

   데이터베이스는 rig 당 센서 1개로 frames 와 images 를 같은 순서로 만들므로
   **frame_id == image_id** 다. 그 번호에 맞춰 다시 쓴다.

출력: <out>/{rigs,cameras,images,frames,points3D}.txt
"""
import argparse, os, sqlite3, shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--keep", required=True, help="고정할 실촬영 파일명 목록")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    keep = {l.strip() for l in open(a.keep) if l.strip()}
    con = sqlite3.connect(a.db)
    dbid = {n: i for i, n in con.execute("select image_id, name from images")}
    dbcam = {n: c for n, c in con.execute("select name, camera_id from images")}
    missing = keep - set(dbid)
    if missing:
        print("  ⚠️ 데이터베이스에 없는 실촬영 %d장은 건너뛴다" % len(missing))
        keep -= missing
    os.makedirs(a.out, exist_ok=True)

    lines = [l for l in open(os.path.join(a.merged, "images.txt")) if not l.startswith("#")]
    old2new, recs = {}, []
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        if len(t) < 10 or t[9] not in keep:
            continue
        old2new[int(t[0])] = dbid[t[9]]
        recs.append((dbid[t[9]], t[1:8], dbcam[t[9]], t[9],
                     lines[i+1] if i+1 < len(lines) else "\n"))
    recs.sort()
    print("  고정 이미지 %d장 (DB ID 로 재번호)" % len(recs))

    # rigs / cameras — 데이터베이스 기준
    with open(os.path.join(a.out, "rigs.txt"), "w") as f:
        f.write("# Rig list\n")
        for rid, rs, rt in con.execute("select rig_id, ref_sensor_id, ref_sensor_type from rigs"):
            f.write("%d 1 CAMERA %d\n" % (rid, rs))
    with open(os.path.join(a.out, "cameras.txt"), "w") as f:
        f.write("# Camera list\n")
        for cid, model, w, h, params in con.execute(
                "select camera_id, model, width, height, params from cameras"):
            import struct
            n = len(params)//8
            p = struct.unpack("<%dd" % n, params)
            f.write("%d OPENCV %d %d %s\n" % (cid, w, h, " ".join("%.10f" % x for x in p)))

    # images / frames — frame_id == image_id
    with open(os.path.join(a.out, "images.txt"), "w") as fi, \
         open(os.path.join(a.out, "frames.txt"), "w") as ff:
        fi.write("# Image list\n")
        ff.write("# Frame list\n")
        for nid, qt, cam, name, pts in recs:
            fi.write("%d %s %d %s\n" % (nid, " ".join(qt), cam, name))
            # 트랙의 점 번호는 그대로 두고, 이미지 번호만 바뀌므로 2행은 그대로 쓴다
            fi.write(pts if pts.endswith("\n") else pts + "\n")
            ff.write("%d 1 %s 1 CAMERA %d %d\n" % (nid, " ".join(qt), cam, nid))

    # points3D — 트랙의 이미지 번호를 새 번호로
    n = 0
    with open(os.path.join(a.out, "points3D.txt"), "w") as f:
        f.write("# 3D point list\n")
        for l in open(os.path.join(a.merged, "points3D.txt")):
            if l.startswith("#"):
                continue
            v = l.split()
            if len(v) < 9:
                continue
            tr = [(int(v[k]), int(v[k+1])) for k in range(8, len(v)-1, 2)]
            tr = [(old2new[i], j) for i, j in tr if i in old2new]
            if len(tr) < 2:
                continue
            f.write(" ".join(v[:8]) + " " + " ".join("%d %d" % x for x in tr) + "\n")
            n += 1
    print("  3D 점 %d개" % n)


if __name__ == "__main__":
    main()
