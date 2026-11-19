import time, torch, gsplat
print("gsplat", gsplat.__version__, "| torch", torch.__version__, flush=True)
dev = "cuda"
N, C, W, H = 2000, 1, 320, 240
g = torch.Generator(device=dev).manual_seed(0)
means    = torch.randn(N, 3, device=dev, generator=g)
quats    = torch.randn(N, 4, device=dev, generator=g)
scales   = torch.rand(N, 3, device=dev, generator=g) * 0.1
opac     = torch.rand(N, device=dev, generator=g)
colors   = torch.rand(N, 3, device=dev, generator=g)
for t in (means, quats, scales, opac, colors): t.requires_grad_(True)
viewmats = torch.eye(4, device=dev)[None].repeat(C, 1, 1)
viewmats[:, 2, 3] = 8.0
f = 300.0
Ks = torch.tensor([[[f,0,W/2],[0,f,H/2],[0,0,1]]], device=dev)

t0 = time.time()
img, alpha, meta = gsplat.rasterization(means, quats, scales, opac, colors,
                                        viewmats, Ks, W, H)
torch.cuda.synchronize()
print(f"[1] 커널 컴파일+첫 렌더: {time.time()-t0:.1f}s  img={tuple(img.shape)} alpha={tuple(alpha.shape)}", flush=True)
print(f"    픽셀 범위 {img.min().item():.4f} ~ {img.max().item():.4f}  평균 alpha {alpha.mean().item():.4f}", flush=True)

t0 = time.time(); img.sum().backward(); torch.cuda.synchronize()
print(f"[2] backward: {time.time()-t0:.2f}s", flush=True)
for n, t in [("means",means),("quats",quats),("scales",scales),("opacities",opac),("colors",colors)]:
    ok = t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
    print(f"    {'OK ' if ok else 'FAIL'} {n:10s} |grad|={0 if t.grad is None else t.grad.abs().sum().item():.4e}", flush=True)

torch.cuda.synchronize(); t0 = time.time(); n_it = 50
for _ in range(n_it):
    gsplat.rasterization(means.detach(), quats.detach(), scales.detach(),
                         opac.detach(), colors.detach(), viewmats, Ks, W, H)
torch.cuda.synchronize()
fps = n_it / (time.time()-t0)
print(f"[3] 렌더 속도: {fps:.0f} FPS  ({W}x{H}, N={N})", flush=True)
print("GSPLAT OK", flush=True)
