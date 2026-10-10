import sys, os, pickle, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'edm'))
net = pickle.load(open('checkpoints/edm-ffhq-64x64-uncond-vp.pkl', 'rb'))['ema'].cuda().eval()
net.use_fp16 = True
torch.backends.cudnn.benchmark = True
x = torch.randn(64, 3, 64, 64, device='cuda'); s = torch.ones(64, device='cuda')
with torch.no_grad():
    for _ in range(3): net(x, s)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(25): net(x, s)
    torch.cuda.synchronize()
print(f'fp16 batch64 cudnn.benchmark: {1000*(time.time()-t0)/25:.0f} ms/NFE = {1000*(time.time()-t0)/25/64:.1f} ms per image-NFE')
